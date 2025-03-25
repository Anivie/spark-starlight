use crate::engine::inference_engine::{ExecutionProvider, OnnxSession};
use crate::inference::sam::SamEncoderOutput;
use crate::utils::graph::SamPrompt;
use crate::utils::tensor::linear_interpolate;
use crate::{INFERENCE_SAM, RUNNING_SAM_DEVICE};
use anyhow::Result;
use bitvec::prelude::*;
use cudarc::driver::{CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
use log::info;
use ndarray::prelude::*;
use ort::inputs;
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::{Session, SessionInputValue, SessionOutputs};
use ort::tensor::Shape;
use ort::value::{DynValue, Tensor, TensorRef, TensorRefMut};
use parking_lot::{Mutex, RwLock};
use spark_media::filter::filter::AVFilter;
use spark_media::Image;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::sync::{Arc, LazyLock};

pub trait SamImageInference {
    fn encode_image(&self, image: Image) -> Result<SamEncoderOutput>;
    fn inference_frame(
        &self,
        prompt: SamPrompt<f32>,
        output_size: Option<(i32, i32)>,
        encoded_result: &SamEncoderOutput,
    ) -> Result<BitVec>;
}

pub struct SAMImageInferenceSession {
    pub(super) image_encoder: Mutex<OnnxSession>,
    pub(super) image_decoder: Mutex<OnnxSession>,
}

impl SAMImageInferenceSession {
    pub fn new(folder_path: impl AsRef<Path>) -> Result<Self> {
        let image_encoder = OnnxSession::new(
            folder_path.as_ref().join("image_encoder.onnx"),
            ExecutionProvider::CPU,
        )?;
        let image_decoder = OnnxSession::new(
            folder_path.as_ref().join("image_decoder.onnx"),
            ExecutionProvider::CPU,
        )?;
        info!("SAM Image Inference Session created");

        Ok(Self {
            image_encoder: Mutex::new(image_encoder),
            image_decoder: Mutex::new(image_decoder),
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
        output_size: Option<(i32, i32)>,
        encoded_result: &SamEncoderOutput,
    ) -> Result<BitVec> {
        let mask_decoder_output = self.inference_image_decoder(
            encoded_result.origin_size,
            encoded_result.encoder_output["image_embeddings"]
                .try_extract_array::<f32>()?
                .into_dimensionality::<Ix4>()?,
            encoded_result.encoder_output["high_res_features1"]
                .try_extract_array::<f32>()?
                .into_dimensionality::<Ix4>()?,
            encoded_result.encoder_output["high_res_features2"]
                .try_extract_array::<f32>()?
                .into_dimensionality::<Ix4>()?,
            &prompt,
        )?;

        let pred_mask = mask_decoder_output["masks"].try_extract_array::<f32>()?;
        let iou_predictions = mask_decoder_output["iou_predictions"].try_extract_array::<f32>()?;
        let max_index = iou_predictions
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        let pred_mask = pred_mask.slice(s![0, max_index, .., ..]);
        let pred_mask = pred_mask.into_shape_with_order((640, 640))?.to_owned();
        let pred_mask = if let Some(size) = output_size {
            info!("Resizing mask from 640x640 to {}x{}", size.0, size.1);
            linear_interpolate(pred_mask, (size.0 as usize, size.1 as usize))
        } else if encoded_result.origin_size.0 != 640 || encoded_result.origin_size.1 != 640 {
            info!(
                "Resizing mask from 640x640 to {}x{}",
                encoded_result.origin_size.0, encoded_result.origin_size.1
            );
            linear_interpolate(
                pred_mask,
                (
                    encoded_result.origin_size.0 as usize,
                    encoded_result.origin_size.1 as usize,
                ),
            )
        } else {
            pred_mask
        };
        info!("now pred_mask shape: {:?}", pred_mask.shape());

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
    fn inference_image_encoder(&self, image: &Vec<u8>) -> Result<HashMap<String, DynValue>> {
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

        let mut image_encoder = self.image_encoder.lock();
        let session_outputs = match image_encoder.executor {
            ExecutionProvider::CPU => {
                let tensor = INFERENCE_SAM.dtoh_sync_copy(&tensor)?;
                let tensor = Array4::from_shape_vec((1, 3, 1024, 1024), tensor)?;
                image_encoder.run(inputs![Tensor::from_array(tensor)?])?
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
                    Shape::new([1, 3, 1024, 1024]),
                )?;
                image_encoder.run(vec![("input_image", SessionInputValue::from(tensor))])?
            },
        };

        let session_outputs = session_outputs
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect::<HashMap<String, DynValue>>();

        Ok(session_outputs)
    }

    fn inference_image_decoder(
        &self,
        image_size: (i32, i32),

        feats: ArrayView4<f32>,
        feat0: ArrayView4<f32>,
        feat1: ArrayView4<f32>,

        prompt: &SamPrompt<f32>,
    ) -> Result<HashMap<String, DynValue>> {
        let mask_input = Array4::<f32>::zeros((1, 1, 256, 256));

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
            SamPrompt::Point(_) => array![[1_f32]],
            SamPrompt::Box(_) => array![[1_f32, 1_f32]],
            SamPrompt::Both(_, _) => array![[1_f32, 1_f32, 1_f32]],
        };

        let prompt_out = match prompt {
            SamPrompt::Point(_) => trans_prompt.into_shape_with_order((1, 1, 2))?,
            SamPrompt::Box(_) => trans_prompt.into_shape_with_order((1, 2, 2))?,
            SamPrompt::Both(_, _) => trans_prompt.into_shape_with_order((1, 3, 2))?,
        };

        let mut decoder = self.image_decoder.lock();
        let result = decoder.run(inputs![
            "image_embeddings"      => TensorRef::from_array_view(feats)?,
            "high_res_features1"    => TensorRef::from_array_view(feat0)?,
            "high_res_features2"    => TensorRef::from_array_view(feat1)?,

            "point_coords"          => Tensor::from_array(prompt_out)?,
            "point_labels"          => Tensor::from_array(point_labels)?,
            "mask_input"            => Tensor::from_array(mask_input)?,
            "has_mask_input"        => Tensor::from_array(array![0_f32])?,

            "orig_im_size"          => Tensor::from_array(array![640i64, 640i64])?,
        ])?;

        let result = result
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect::<HashMap<String, DynValue>>();

        Ok(result)
    }
}
