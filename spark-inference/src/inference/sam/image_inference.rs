use crate::engine::inference_engine::{ExecutionProvider, OnnxSession};
use crate::inference::sam::video_inference::SamEncoderOutput;
use crate::utils::graph::SamPrompt;
use crate::utils::tensor::linear_interpolate;
use crate::{INFERENCE_SAM, RUNNING_SAM_DEVICE};
use anyhow::Result;
use bitvec::prelude::*;
use cudarc::driver::{CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
use ndarray::prelude::*;
use ort::inputs;
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::{SessionInputValue, SessionOutputs};
use ort::value::{Tensor, TensorRef, TensorRefMut};
use spark_media::filter::filter::AVFilter;
use spark_media::Image;
use std::ops::Deref;
use std::path::Path;
use std::sync::LazyLock;

pub trait SamImageInference {
    fn encode_image(&self, image: Image) -> Result<SamEncoderOutput>;
    fn inference_frame(
        &self,
        prompt: SamPrompt<f32>,
        encoded_result: &SamEncoderOutput,
    ) -> Result<BitVec>;
}

pub struct SAMImageInferenceSession {
    pub(super) image_encoder: OnnxSession,
    pub(super) mask_decoder: OnnxSession,
    pub(super) prompt_encoder: OnnxSession,
}

impl SAMImageInferenceSession {
    pub fn new(folder_path: impl AsRef<Path>) -> Result<Self> {
        let image_encoder = OnnxSession::new(
            folder_path.as_ref().join("image_encoder.onnx"),
            ExecutionProvider::CPU,
        )?;
        let mask_decoder = OnnxSession::new(
            folder_path.as_ref().join("mask_decoder.onnx"),
            ExecutionProvider::CPU,
        )?;
        let prompt_encoder = OnnxSession::new(
            folder_path.as_ref().join("prompt_encoder.onnx"),
            ExecutionProvider::CPU,
        )?;

        Ok(Self {
            image_encoder,
            mask_decoder,
            prompt_encoder,
        })
    }
}

impl SamImageInference for SAMImageInferenceSession {
    fn encode_image(&self, mut image: Image) -> Result<SamEncoderOutput> {
        let (image, image_size) = {
            let filter = AVFilter::builder(image.pixel_format()?, image.get_size())?
                .add_context("scale", "1024:1024")?
                .add_context("format", "rgb24")?
                .build()?;

            let image_size = image.get_size();
            image.apply_filter(&filter)?;
            (image, image_size)
        };

        Ok(SamEncoderOutput {
            encoder_output: self.inference_image_encoder(image.raw_data()?.deref())?,
            origin_size: image_size,
        })
    }

    fn inference_frame(
        &self,
        prompt: SamPrompt<f32>,
        encoded_result: &SamEncoderOutput,
    ) -> Result<BitVec> {
        let mask_decoder_output = self.inference_image_decoder(
            encoded_result.origin_size,
            encoded_result.encoder_output["vision_features"]
                .try_extract_tensor::<f32>()?
                .into_dimensionality::<Ix4>()?,
            encoded_result.encoder_output["backbone_fpn_0"]
                .try_extract_tensor::<f32>()?
                .into_dimensionality::<Ix4>()?,
            encoded_result.encoder_output["backbone_fpn_1"]
                .try_extract_tensor::<f32>()?
                .into_dimensionality::<Ix4>()?,
            &prompt,
        )?;

        let pred_mask = mask_decoder_output["masks"].try_extract_tensor::<f32>()?;
        let iou_predictions = mask_decoder_output["iou_pred"].try_extract_tensor::<f32>()?;
        let max_index = iou_predictions
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        let pred_mask = pred_mask.slice(s![0, max_index, .., ..]);
        let pred_mask = pred_mask.into_shape_with_order((256, 256))?;
        let pred_mask = linear_interpolate(pred_mask.into_owned(), (1024, 1024));

        let back = {
            let mut back = BitVec::with_capacity(pred_mask.len());
            pred_mask.iter().for_each(|x| {
                back.push(*x > 0f32);
            });
            back
        };
        Ok(back)
    }
}

impl SAMImageInferenceSession {
    fn inference_image_encoder(&self, image: &Vec<u8>) -> Result<SessionOutputs> {
        static MEAN: LazyLock<CudaSlice<f32>> = LazyLock::new(|| {
            INFERENCE_SAM
                .htod_sync_copy(&[0.485, 0.456, 0.406])
                .unwrap()
        });
        static STD: LazyLock<CudaSlice<f32>> = LazyLock::new(|| {
            INFERENCE_SAM
                .htod_sync_copy(&[0.229, 0.224, 0.225])
                .unwrap()
        });

        let buffer = INFERENCE_SAM.htod_sync_copy(image.as_slice())?;
        let cfg = LaunchConfig::for_num_elems((image.len() / 3) as u32);

        let tensor = unsafe {
            let mut tensor = INFERENCE_SAM.alloc::<f32>(image.len())?;

            INFERENCE_SAM.normalise_pixel_mean().launch(
                cfg,
                (&mut tensor, &buffer, MEAN.deref(), STD.deref(), image.len()),
            )?;

            tensor
        };

        let session_outputs = match self.image_encoder.executor {
            ExecutionProvider::CPU => {
                let tensor = INFERENCE_SAM.dtoh_sync_copy(&tensor)?;
                let tensor = Array4::from_shape_vec((1, 3, 1024, 1024), tensor)?;
                self.image_encoder
                    .run(inputs![Tensor::from_array(tensor)?])?
            }
            _ => unsafe {
                let tensor: TensorRefMut<'_, f32> = TensorRefMut::from_raw(
                    MemoryInfo::new(
                        AllocationDevice::CUDA,
                        RUNNING_SAM_DEVICE,
                        AllocatorType::Device,
                        MemoryType::Default,
                    )?,
                    (*tensor.device_ptr() as usize as *mut ()).cast(),
                    vec![1, 3, 1024, 1024],
                )?;
                self.image_encoder
                    .run(vec![("input_image", SessionInputValue::from(tensor))])?
            },
        };

        Ok(session_outputs)
    }

    fn inference_image_decoder(
        &self,
        image_size: (i32, i32),

        feats: ArrayView4<f32>,
        feat0: ArrayView4<f32>,
        feat1: ArrayView4<f32>,

        prompt: &SamPrompt<f32>,
    ) -> Result<SessionOutputs> {
        let mask_input = Array3::<f32>::zeros((1, 256, 256));

        let trans_prompt = match &prompt {
            SamPrompt::Box(boxes) => {
                array![
                    1024f32 * (boxes.x / image_size.0 as f32),
                    1024f32 * (boxes.y / image_size.1 as f32),
                    1024f32 * ((boxes.x + boxes.width / 2_f32) / image_size.0 as f32),
                    1024f32 * ((boxes.y + boxes.height / 2_f32) / image_size.1 as f32),
                ]
            }
            SamPrompt::Point(point) => {
                array![
                    1024f32 * (point.x / image_size.0 as f32),
                    1024f32 * (point.y / image_size.1 as f32),
                ]
            }
            SamPrompt::Both(point, boxes) => {
                array![
                    1024f32 * (point.x / image_size.0 as f32),
                    1024f32 * (point.y / image_size.1 as f32),
                    1024f32 * (boxes.x / image_size.0 as f32),
                    1024f32 * (boxes.y / image_size.1 as f32),
                    1024f32 * ((boxes.x + boxes.width / 2_f32) / image_size.0 as f32),
                    1024f32 * ((boxes.y + boxes.height / 2_f32) / image_size.1 as f32),
                ]
            }
        };

        let point_labels = match prompt {
            SamPrompt::Point(_) => array![[1]],
            SamPrompt::Box(_) => array![[1, 1]],
            SamPrompt::Both(_, _) => array![[1, 1, 1]],
        };

        let prompt_out = match prompt {
            SamPrompt::Point(_) => trans_prompt.into_shape_with_order((1, 1, 2))?,
            SamPrompt::Box(_) => trans_prompt.into_shape_with_order((1, 2, 2))?,
            SamPrompt::Both(_, _) => trans_prompt.into_shape_with_order((1, 3, 2))?,
        };

        let prompt_result = self.prompt_encoder.run(inputs![
            "coords" => Tensor::from_array(prompt_out)?,
            "labels" => Tensor::from_array(point_labels)?,
            "masks" => Tensor::from_array(mask_input)?,
            "masks_enable" => Tensor::from_array(array![0])?,
        ])?;

        let result = self.mask_decoder.run(inputs![
            "image_embeddings" => TensorRef::from_array_view(feats)?,
            "high_res_features1" => TensorRef::from_array_view(feat0)?,
            "high_res_features2" => TensorRef::from_array_view(feat1)?,

            "image_pe" => TensorRef::from_array_view(prompt_result["dense_pe"].try_extract_tensor::<f32>()?)?,
            "sparse_prompt_embeddings" => TensorRef::from_array_view(prompt_result["sparse_embeddings"].try_extract_tensor::<f32>()?)?,
            "dense_prompt_embeddings" => TensorRef::from_array_view(prompt_result["dense_embeddings"].try_extract_tensor::<f32>()?)?,
        ])?;

        Ok(result)
    }
}
