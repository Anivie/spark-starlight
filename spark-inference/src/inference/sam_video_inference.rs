use crate::engine::inference_engine::{ExecutionProvider, OnnxSession};
use crate::inference::sam_former_state::{FormerState, InferenceResult, InferenceType};
use crate::utils::graph::Point;
use crate::INFERENCE_CUDA;
use anyhow::Result;
use bitvec::prelude::*;
use cudarc::driver::{CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
use ndarray::{array, Array1, Array2, Array3, Array4, ArrayViewD};
use spark_media::filter::filter::AVFilter;
use spark_media::Image;
use std::ops::Deref;
use std::path::Path;
use std::sync::LazyLock;
use ort::inputs;
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::{SessionInputValue, SessionOutputs};
use ort::value::TensorRefMut;

pub trait SamVideoInference {
    fn inference_sam(&self, types: InferenceType, image: &mut Image) -> Result<InferenceResult>;
}

pub struct SAM2VideoInferenceSession {
    image_encoder: OnnxSession,
    image_decoder: OnnxSession,
    memory_attention: OnnxSession,
    memory_encoder: OnnxSession,
}

impl SAM2VideoInferenceSession {
    pub fn new(folder_path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            image_encoder: OnnxSession::new(folder_path.as_ref().join("image_encoder.onnx"), ExecutionProvider::CUDA)?,
            image_decoder: OnnxSession::new(folder_path.as_ref().join("image_decoder.onnx"), ExecutionProvider::CUDA)?,
            memory_attention: OnnxSession::new(folder_path.as_ref().join("memory_attention.onnx"), ExecutionProvider::CUDA)?,
            memory_encoder: OnnxSession::new(folder_path.as_ref().join("memory_encoder.onnx"), ExecutionProvider::CUDA)?,
        })
    }
}

impl SamVideoInference for SAM2VideoInferenceSession {
    fn inference_sam(&self, types: InferenceType, image: &mut Image) -> Result<InferenceResult> {
        let filter = AVFilter::builder(image.pixel_format()?, image.get_size())?
            .add_context("scale", "1024:1024")?
            .add_context("format", "rgb24")?
            .build()?;

        let image_size = image.get_size();
        image.apply_filter(&filter)?;

        let encoder_output = self.inference_image_encoder(image.raw_data()?.deref())?;

        let mut memory_attention_output = None;
        let vision_feats = {
            match &types {
                InferenceType::First(_) => encoder_output["vision_feats"].try_extract_tensor::<f32>()?,
                InferenceType::WithState(state) => {
                    println!("object_memory: {:?}", state.object_memory().shape());
                    println!("mask_memory: {:?}", state.mask_memory().shape());
                    println!("mask_pos_embed: {:?}", state.mask_pos_embed().shape());
                    let memory_attention_output_inner = self.inference_memory_attention(
                        encoder_output["vision_feats"].try_extract_tensor::<f32>()?.view(),
                        encoder_output["vision_pos_embed"].try_extract_tensor::<f32>()?.view(),
                        state.object_memory(),
                        state.mask_memory(),
                        state.mask_pos_embed(),
                    )?;

                    memory_attention_output.replace(memory_attention_output_inner);
                    let outputs = memory_attention_output.as_ref().unwrap();
                    outputs["image_embed"].try_extract_tensor::<f32>()?
                }
            }
        };

        let decoder_output = self.inference_image_decoder(
            image_size,
            vision_feats,
            encoder_output["high_res_feat0"].try_extract_tensor::<f32>()?,
            encoder_output["high_res_feat1"].try_extract_tensor::<f32>()?,
            types.get_points(),
        )?;

        let memory_encoder_output = self.inference_memory_encoder(
            decoder_output["mask_for_mem"].try_extract_tensor::<f32>()?.view(),
            encoder_output["pix_feat"].try_extract_tensor::<f32>()?.view()
        )?;

        let state = match types {
            InferenceType::First(point) => {
                FormerState::new(&decoder_output, &memory_encoder_output, point)?
            }
            InferenceType::WithState(state) => {
                state.update(&decoder_output, &memory_encoder_output)?
            }
        };

        let back = {
            let mut back = BitVec::with_capacity((image.get_width() * image.get_height()) as usize);
            let pred_mask = decoder_output["pred_mask"].try_extract_tensor::<f32>()?;
            pred_mask.iter().for_each(|x| {
                back.push(*x > 0f32);
            });
            back
        };

        Ok(InferenceResult::new(back, state))
    }
}

impl SAM2VideoInferenceSession {
    fn inference_image_encoder(&self, image: &Vec<u8>) -> Result<SessionOutputs> {
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

        INFERENCE_CUDA.synchronize()?;
        Ok(self.image_encoder.run(vec![("image", SessionInputValue::from(tensor))])?)
    }

    fn inference_memory_attention(
        &self,
        vision_feats: ArrayViewD<f32>, //Every vision_feats for encoder output in each round
        vision_pos_embed: ArrayViewD<f32>, //Every vision_pos_embed for encoder output in each round

        object_memory: &Array2<f32>, //Every obj_ptr for decoder output in each round
        mask_memory: &Array4<f32>, //Every maskmem_features for memory encoder output in each round
        mask_position_embedded: &Array3<f32>, //Every maskmem_pos_enc + temporal_code for decoder output in each round
    ) -> Result<SessionOutputs> {
        let result = self.memory_attention.run(inputs![
            "current_vision_feat" => vision_feats,
            "current_vision_pos_embed" => vision_pos_embed,
            "memory_0" => object_memory.clone(),
            "memory_1" => mask_memory.clone(),
            "memory_pos_embed" => mask_position_embedded.clone(),
        ]?)?;

        Ok(result)
    }

    fn inference_memory_encoder(
        &self,
        mask_for_mem: ArrayViewD<f32>,
        pix_feat: ArrayViewD<f32>,
    ) -> Result<SessionOutputs> {
        let result = self.memory_encoder.run(inputs![
            "mask_for_mem" => mask_for_mem,
            "pix_feat" => pix_feat,
        ]?)?;

        Ok(result)
    }

    fn inference_image_decoder(
        &self,
        image_size: (i32, i32),

        feats: ArrayViewD<f32>,
        feat0: ArrayViewD<f32>,
        feat1: ArrayViewD<f32>,

        points: &Vec<Point<u32>>,
    ) -> Result<SessionOutputs> {
        let point_labels = Array2::from_shape_vec(
            (1, points.len()), vec![1_f32; points.len()]
        )?;

        let points = points
            .iter()
            .map(|point| array![
                1024f32 * (point.x as f32 / image_size.0 as f32),
                1024f32 * (point.y as f32 / image_size.1 as f32),
            ])
            .collect::<Vec<Array1<f32>>>();

        let points = Array3::from_shape_vec(
            (1, points.len(), 2), points.into_iter().flatten().collect()
        )?;

        let image_size = array![image_size.1 as i64, image_size.0 as i64];

        let result = self.image_decoder.run(inputs![
            "point_coords" => points,
            "point_labels" => point_labels,
            "frame_size" => image_size,

            "image_embed" => feats,
            "high_res_feats_0" => feat0,
            "high_res_feats_1" => feat1,
        ]?)?;

        Ok(result)
    }
}