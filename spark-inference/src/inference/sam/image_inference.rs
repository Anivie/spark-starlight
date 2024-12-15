use crate::engine::inference_engine::{ExecutionProvider, OnnxSession};
use crate::utils::graph::BoxOrPoint;
use crate::INFERENCE_CUDA;
use anyhow::Result;
use bitvec::prelude::*;
use cudarc::driver::{CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
use ndarray::{array, Array2, ArrayViewD};
use ort::inputs;
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::{SessionInputValue, SessionOutputs};
use ort::value::TensorRefMut;
use spark_media::filter::filter::AVFilter;
use spark_media::Image;
use std::ops::Deref;
use std::path::Path;
use std::sync::LazyLock;

pub trait SamImageInference {
    fn inference_sam(&self, prompt: BoxOrPoint<f32>, image: Image) -> Result<BitVec>;
    fn encode_image(&self, image: Image) -> Result<SamEncoderOutput>;
    fn decode_image(&self, prompt: BoxOrPoint<f32>, encoded_result: &SamEncoderOutput) -> Result<BitVec>;
}

pub struct SAM2ImageInferenceSession {
    image_encoder: OnnxSession,
    image_decoder: OnnxSession,
}

pub struct SamEncoderOutput<'a> {
    encoder_output: SessionOutputs<'a, 'a>,

    origin_size: (i32, i32),
}

impl SAM2ImageInferenceSession {
    pub fn new(folder_path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            image_encoder: OnnxSession::new(folder_path.as_ref().join("image_encoder.onnx"), ExecutionProvider::CUDA)?,
            image_decoder: OnnxSession::new(folder_path.as_ref().join("image_decoder.onnx"), ExecutionProvider::CUDA)?,
        })
    }

    pub(super) fn raw(image_encoder: OnnxSession, image_decoder: OnnxSession) -> Self {
        Self {
            image_encoder,
            image_decoder,
        }
    }
}

impl SamImageInference for SAM2ImageInferenceSession {
    fn inference_sam(&self, prompt: BoxOrPoint<f32>, mut image: Image) -> Result<BitVec> {
        let (image, image_size) = {
            let filter = AVFilter::builder(image.pixel_format()?, image.get_size())?
                .add_context("scale", "1024:1024")?
                .add_context("format", "rgb24")?
                .build()?;

            let image_size = image.get_size();
            image.apply_filter(&filter)?;
            (image, image_size)
        };

        let encoder_output = self.inference_image_encoder(image.raw_data()?.deref())?;

        let decoder_output = self.inference_image_decoder(
            image_size,
            encoder_output["vision_feats"].try_extract_tensor::<f32>()?,
            encoder_output["high_res_feat0"].try_extract_tensor::<f32>()?,
            encoder_output["high_res_feat1"].try_extract_tensor::<f32>()?,
            prompt,
        )?;

        let back = {
            let pred_mask = decoder_output["pred_mask"].try_extract_tensor::<f32>()?;
            let mut back = BitVec::with_capacity(pred_mask.len());
            pred_mask.iter().for_each(|x| {
                back.push(*x > 0f32);
            });
            back
        };

        Ok(back)
    }

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

    fn decode_image(&self, prompt: BoxOrPoint<f32>, encoded_result: &SamEncoderOutput) -> Result<BitVec> {
        let decoder_output = self.inference_image_decoder(
            encoded_result.origin_size,
            encoded_result.encoder_output["vision_feats"].try_extract_tensor::<f32>()?,
            encoded_result.encoder_output["high_res_feat0"].try_extract_tensor::<f32>()?,
            encoded_result.encoder_output["high_res_feat1"].try_extract_tensor::<f32>()?,
            prompt,
        )?;

        let pred_mask = decoder_output["pred_mask"].try_extract_tensor::<f32>()?;

        let mut back = BitVec::with_capacity(pred_mask.len());

        pred_mask.iter().for_each(|x| {
            back.push(*x > 0f32);
        });

        Ok(back)
    }
}

impl SAM2ImageInferenceSession {
    pub(crate) fn inference_image_encoder(&self, image: &Vec<u8>) -> Result<SessionOutputs> {
        static MEAN: LazyLock<CudaSlice<f32>> = LazyLock::new(|| {
            INFERENCE_CUDA.htod_sync_copy(&[0.485, 0.456, 0.406]).unwrap()
        });
        static STD: LazyLock<CudaSlice<f32>> = LazyLock::new(|| {
            INFERENCE_CUDA.htod_sync_copy(&[0.229, 0.224, 0.225]).unwrap()
        });

        let buffer = INFERENCE_CUDA.htod_sync_copy(image.as_slice())?;
        let cfg = LaunchConfig::for_num_elems((image.len() / 3) as u32);

        let tensor: TensorRefMut<'_, f32> = unsafe {
            let mut tensor = INFERENCE_CUDA.alloc::<f32>(image.len())?;

            INFERENCE_CUDA.normalise_pixel_mean().launch(cfg, (
                &mut tensor, &buffer,
                MEAN.deref(), STD.deref(),
                image.len(),
            ))?;

            TensorRefMut::from_raw(
                MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default)?,
                (*tensor.device_ptr() as usize as *mut ()).cast(),
                vec![1, 3, 1024, 1024],
            )?
        };

        Ok(self.image_encoder.run(vec![("image", SessionInputValue::from(tensor))])?)
    }

    pub(crate) fn inference_image_decoder(
        &self,
        image_size: (i32, i32),

        feats: ArrayViewD<f32>,
        feat0: ArrayViewD<f32>,
        feat1: ArrayViewD<f32>,

        prompt: BoxOrPoint<f32>,
    ) -> Result<SessionOutputs> {
        let trans_prompt = match &prompt {
            BoxOrPoint::Box(point) => {
                array![
                        1024f32 * (point.x / image_size.0 as f32),
                        1024f32 * (point.y / image_size.1 as f32),
                        1024f32 * ((point.x + point.width / 2_f32) / image_size.0 as f32),
                        1024f32 * ((point.y + point.height / 2_f32) / image_size.1 as f32),
                    ]
            }
            BoxOrPoint::Point(point) => {
                array![
                        1024f32 * (point.x / image_size.0 as f32),
                        1024f32 * (point.y / image_size.1 as f32),
                    ]
            }
        };

        let point_labels = match prompt {
            BoxOrPoint::Point(_) => Array2::from_shape_vec(
                (1, 1), vec![1_f32]
            )?,
            BoxOrPoint::Box(_) => Array2::from_shape_vec(
                (1, 2), vec![2_f32, 3_f32]
            )?,
        };

        let image_size = array![image_size.1 as i64, image_size.0 as i64];

        let prompt = match prompt {
            BoxOrPoint::Point(_) => trans_prompt.into_shape_with_order((1, 1, 2))?,
            BoxOrPoint::Box(_) => trans_prompt.into_shape_with_order((1, 2, 2))?,
        };

        let result = self.image_decoder.run(inputs![
            "point_coords" => prompt,
            "point_labels" => point_labels,
            "frame_size" => image_size,

            "image_embed" => feats,
            "high_res_feats_0" => feat0,
            "high_res_feats_1" => feat1,
        ]?)?;

        Ok(result)
    }
}