use crate::engine::inference_engine::{ExecutionProvider, OnnxSession};
use crate::inference::linear_interpolate;
use crate::utils::graph::SamPrompt;
use crate::INFERENCE_CUDA;
use anyhow::Result;
use bitvec::prelude::*;
use cudarc::driver::{CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
use ndarray::{array, s, Array, Array2, ArrayBase, ArrayViewD, OwnedRepr};
use ort::inputs;
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::{SessionInputValue, SessionOutputs};
use ort::value::TensorRefMut;
use spark_media::filter::filter::AVFilter;
use spark_media::Image;
use std::ops::Deref;
use std::path::Path;
use std::sync::LazyLock;

pub trait SamVideoInference {
    fn encode_image(&self, image: Image) -> Result<SamEncoderOutput>;
    fn decode_image(
        &self,
        prompt: SamPrompt<f32>,
        encoded_result: &SamEncoderOutput,
    ) -> Result<BitVec>;
}

pub struct SAMVideoInferenceSession {
    image_encoder: OnnxSession,
    mask_decoder: OnnxSession,
    prompt_encoder: OnnxSession,

    mlp: OnnxSession,
    memory_encoder: OnnxSession,
    memory_attention: OnnxSession,
    obj_ptr_tpos_proj: OnnxSession,
}

pub struct SamEncoderOutput<'a> {
    encoder_output: SessionOutputs<'a, 'a>,

    origin_size: (i32, i32),
}

impl SAMVideoInferenceSession {
    pub fn new(folder_path: impl AsRef<Path>) -> Result<Self> {
        let image_encoder = OnnxSession::new(
            folder_path.as_ref().join("image_encoder.onnx"),
            ExecutionProvider::CUDA,
        )?;
        let mask_decoder = OnnxSession::new(
            folder_path.as_ref().join("mask_decoder.onnx"),
            ExecutionProvider::CUDA,
        )?;
        let prompt_encoder = OnnxSession::new(
            folder_path.as_ref().join("prompt_encoder.onnx"),
            ExecutionProvider::CUDA,
        )?;
        let mlp = OnnxSession::new(
            folder_path.as_ref().join("mlp.onnx"),
            ExecutionProvider::CUDA,
        )?;
        let memory_encoder = OnnxSession::new(
            folder_path.as_ref().join("memory_encoder.onnx"),
            ExecutionProvider::CUDA,
        )?;
        let memory_attention = OnnxSession::new(
            folder_path.as_ref().join("memory_attention.onnx"),
            ExecutionProvider::CUDA,
        )?;
        let obj_ptr_tpos_proj = OnnxSession::new(
            folder_path.as_ref().join("obj_ptr_tpos_proj.onnx"),
            ExecutionProvider::CUDA,
        )?;
        println!("mlp input: {:?}, output: {:?}", mlp.inputs, mlp.outputs);
        println!(
            "memory_encoder input: {:?}, output: {:?}",
            memory_encoder.inputs, memory_encoder.outputs
        );
        println!(
            "memory_attention input: {:?}, output: {:?}",
            memory_attention.inputs, memory_attention.outputs
        );
        println!(
            "obj_ptr_tpos_proj input: {:?}, output: {:?}",
            obj_ptr_tpos_proj.inputs, obj_ptr_tpos_proj.outputs
        );

        Ok(Self {
            image_encoder,
            mask_decoder,
            prompt_encoder,
            mlp,
            memory_encoder,
            memory_attention,
            obj_ptr_tpos_proj,
        })
    }
}

impl SamVideoInference for SAMVideoInferenceSession {
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

    fn decode_image(
        &self,
        prompt: SamPrompt<f32>,
        encoded_result: &SamEncoderOutput,
    ) -> Result<BitVec> {
        let decoder_output = self.inference_image_decoder(
            encoded_result.origin_size,
            encoded_result.encoder_output["vision_features"].try_extract_tensor::<f32>()?,
            encoded_result.encoder_output["backbone_fpn_0"].try_extract_tensor::<f32>()?,
            encoded_result.encoder_output["backbone_fpn_1"].try_extract_tensor::<f32>()?,
            prompt,
        )?;

        let pred_mask = decoder_output["masks"].try_extract_tensor::<f32>()?;
        let iou_predictions = decoder_output["iou_pred"].try_extract_tensor::<f32>()?;
        println!("mask: {:?}, iou: {:?}", pred_mask.shape(), iou_predictions);

        let pred_mask = pred_mask.slice(s![0, 0, .., ..]);
        let pred_mask = pred_mask.into_shape_with_order((256, 256))?;
        let pred_mask = linear_interpolate(pred_mask.into_owned(), (1024, 1024));

        let mut back = BitVec::with_capacity(pred_mask.len());

        pred_mask.iter().for_each(|x| {
            back.push(*x > 0f32);
        });

        Ok(back)
    }
}

impl SAMVideoInferenceSession {
    pub(crate) fn inference_image_encoder(&self, image: &Vec<u8>) -> Result<SessionOutputs> {
        static MEAN: LazyLock<CudaSlice<f32>> = LazyLock::new(|| {
            INFERENCE_CUDA
                .htod_sync_copy(&[0.485, 0.456, 0.406])
                .unwrap()
        });
        static STD: LazyLock<CudaSlice<f32>> = LazyLock::new(|| {
            INFERENCE_CUDA
                .htod_sync_copy(&[0.229, 0.224, 0.225])
                .unwrap()
        });

        let buffer = INFERENCE_CUDA.htod_sync_copy(image.as_slice())?;
        let cfg = LaunchConfig::for_num_elems((image.len() / 3) as u32);

        let tensor: TensorRefMut<'_, f32> = unsafe {
            let mut tensor = INFERENCE_CUDA.alloc::<f32>(image.len())?;

            INFERENCE_CUDA.normalise_pixel_mean().launch(
                cfg,
                (&mut tensor, &buffer, MEAN.deref(), STD.deref(), image.len()),
            )?;

            TensorRefMut::from_raw(
                MemoryInfo::new(
                    AllocationDevice::CUDA,
                    0,
                    AllocatorType::Device,
                    MemoryType::Default,
                )?,
                (*tensor.device_ptr() as usize as *mut ()).cast(),
                vec![1, 3, 1024, 1024],
            )?
        };

        Ok(self
            .image_encoder
            .run(vec![("input_image", SessionInputValue::from(tensor))])?)
    }

    pub(crate) fn inference_image_decoder(
        &self,
        image_size: (i32, i32),

        feats: ArrayViewD<f32>,
        feat0: ArrayViewD<f32>,
        feat1: ArrayViewD<f32>,

        prompt: SamPrompt<f32>,
    ) -> Result<SessionOutputs> {
        let mask_input: ArrayBase<OwnedRepr<f32>, _> = Array::zeros((1, 256, 256));

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
            SamPrompt::Point(_) => Array2::from_shape_vec((1, 1), vec![1])?,
            SamPrompt::Box(_) => Array2::from_shape_vec((1, 2), vec![1, 1])?,
            SamPrompt::Both(_, _) => Array2::from_shape_vec((1, 3), vec![1, 1, 1])?,
        };

        let prompt = match prompt {
            SamPrompt::Point(_) => trans_prompt.into_shape_with_order((1, 1, 2))?,
            SamPrompt::Box(_) => trans_prompt.into_shape_with_order((1, 2, 2))?,
            SamPrompt::Both(_, _) => trans_prompt.into_shape_with_order((1, 3, 2))?,
        };

        let prompt_result = self.prompt_encoder.run(inputs![
            "coords" => prompt,
            "labels" => point_labels,
            "masks" => mask_input,
            "masks_enable" => array![0],
        ]?)?;

        let result = self.mask_decoder.run(inputs![
            "image_embeddings" => feats,
            "high_res_features1" => feat0,
            "high_res_features2" => feat1,

            "image_pe" => prompt_result["dense_pe"].try_extract_tensor::<f32>()?,
            "sparse_prompt_embeddings" => prompt_result["sparse_embeddings"].try_extract_tensor::<f32>()?,
            "dense_prompt_embeddings" => prompt_result["dense_embeddings"].try_extract_tensor::<f32>()?,
        ]?)?;

        Ok(result)
    }
}
