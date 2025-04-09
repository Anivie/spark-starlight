use crate::engine::inference_engine::{ExecutionProvider, OnnxSession};
use crate::utils::graph::SamPrompt;
use crate::utils::tensor::linear_interpolate;
use crate::{INFERENCE_SAM, RUNNING_SAM_DEVICE};
use anyhow::{anyhow, Result};
use bitvec::prelude::*;
use cudarc::driver::{CudaSlice, DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig};
use log::info;
use ndarray::prelude::*;
use ort::inputs;
use ort::io_binding::IoBinding;
use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::session::run_options::OutputSelector;
use ort::session::{HasSelectedOutputs, RunOptions, Session, SessionInputValue, SessionOutputs};
use ort::tensor::Shape;
use ort::value::{DynValue, Tensor, TensorRef, TensorRefMut};
use parking_lot::{Mutex, RwLock};
use spark_media::filter::filter::AVFilter;
use spark_media::Image;
use std::collections::HashMap;
use std::ffi::c_void;
use std::io::Write;
use std::mem::forget;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::sync::{Arc, LazyLock};

pub trait SamImageInference {
    fn inference_frame(
        &self,
        image: Image,
        prompts: Vec<Vec<SamPrompt<f32>>>,
    ) -> Result<Vec<Vec<BitVec>>>;
}

pub struct SAMImageInferenceSession {
    pub(super) image_encoder: Mutex<OnnxSession>,
    pub(super) image_decoder: Mutex<OnnxSession>,
}

impl SAMImageInferenceSession {
    pub fn new(folder_path: impl AsRef<Path>) -> Result<Self> {
        let image_encoder = OnnxSession::new(
            folder_path.as_ref().join("image_encoder.onnx"),
            ExecutionProvider::CUDA(RUNNING_SAM_DEVICE),
        )?;
        let image_decoder = OnnxSession::new(
            folder_path.as_ref().join("image_decoder.onnx"),
            ExecutionProvider::CUDA(RUNNING_SAM_DEVICE),
        )?;
        info!("SAM Image Inference Session created");

        Ok(Self {
            image_encoder: Mutex::new(image_encoder),
            image_decoder: Mutex::new(image_decoder),
        })
    }
}

impl SamImageInference for SAMImageInferenceSession {
    fn inference_frame(
        &self,
        mut image: Image,
        prompts: Vec<Vec<SamPrompt<f32>>>,
    ) -> Result<Vec<Vec<BitVec>>> {
        let (image, image_size) = {
            let filter = AVFilter::builder(image.pixel_format()?, image.get_size())?
                .add_context("scale", "1024:1024")?
                .add_context("format", "rgb24")?
                .build()?;

            let image_size = image.get_size();
            image.apply_filter(&filter)?;
            (image, image_size)
        };

        let mut image_encoder = self.image_encoder.lock();
        let mut encoder_binding = image_encoder.create_binding()?;
        let encoder_output = self.inference_image_encoder(
            image.raw_data()?.deref(),
            &mut encoder_binding,
            image_encoder.deref_mut(),
        )?;

        let mut decoder = self.image_decoder.lock();
        let mut decoder_binding = decoder.create_binding()?;
        decoder_binding.bind_input("image_embed", &encoder_output["image_embed"])?;
        decoder_binding.bind_input("high_res_feats_0", &encoder_output["high_res_feats_0"])?;
        decoder_binding.bind_input("high_res_feats_1", &encoder_output["high_res_feats_1"])?;

        let mut back = vec![];
        for prompt in prompts {
            let mut back_inner = vec![];
            for prompt in prompt {
                let mask_decoder_output = self.inference_image_decoder(
                    &prompt,
                    image_size,
                    &mut decoder_binding,
                    decoder.deref_mut(),
                )?;

                let max_index = {
                    let iou_predictions =
                        mask_decoder_output["iou_predictions"].try_extract_array::<f32>()?;
                    iou_predictions
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .ok_or(anyhow!("No max index found"))?
                        .0
                };

                let pred_mask = {
                    let value = &mask_decoder_output["masks"];
                    let masks = value.data_ptr()?.cast::<f32>();

                    let tensor = unsafe {
                        let mut tensor = INFERENCE_SAM.alloc::<f32>(1024 * 1024)?;
                        let mask =
                            masks.add(image_size.0 as usize * image_size.1 as usize * max_index);
                        let mask = INFERENCE_SAM.upgrade_device_ptr::<f32>(
                            mask as u64,
                            image_size.0 as usize * image_size.1 as usize,
                        );
                        let mask = Box::leak(Box::new(mask));

                        INFERENCE_SAM.bilinear_interpolate_centered().launch(
                            compute_launch_config(1024, 1024, (16, 16, 1)),
                            (mask, &mut tensor, image_size.0, image_size.1, 1024, 1024),
                        )?;
                        tensor
                    };

                    INFERENCE_SAM.dtoh_sync_copy(&tensor)?
                };

                let mut temp = BitVec::with_capacity(pred_mask.len());
                pred_mask.iter().for_each(|x| {
                    temp.push(*x > 0f32);
                });
                back_inner.push(temp);
            }
            back.push(back_inner);
        }

        Ok(back)
    }
}

impl SAMImageInferenceSession {
    fn inference_image_encoder<'a, 'b: 'a>(
        &self,
        image: &Vec<u8>,
        encoder_binding: &'a mut IoBinding,
        image_encoder: &'b mut OnnxSession,
    ) -> Result<SessionOutputs<'a, 'b>> {
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

                encoder_binding.bind_input("image", &tensor)?;

                let allocator = Allocator::new(
                    image_encoder,
                    MemoryInfo::new(
                        AllocationDevice::CUDA_PINNED,
                        RUNNING_SAM_DEVICE,
                        AllocatorType::Device,
                        MemoryType::CPUOutput,
                    )?,
                )?;
                encoder_binding.bind_output_to_device("image_embed", &allocator.memory_info())?;
                encoder_binding
                    .bind_output_to_device("high_res_feats_0", &allocator.memory_info())?;
                encoder_binding
                    .bind_output_to_device("high_res_feats_1", &allocator.memory_info())?;
                image_encoder.run_binding(encoder_binding)?
            },
        };

        Ok(session_outputs)
    }

    fn inference_image_decoder<'a, 'b: 'a>(
        &self,
        prompt: &SamPrompt<f32>,
        image_size: (i32, i32),
        decoder_binding: &'a mut IoBinding,
        decoder: &'b mut OnnxSession,
    ) -> Result<SessionOutputs<'a, 'b>> {
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

        decoder_binding.bind_input("point_coords", &Tensor::from_array(prompt_out)?)?;
        decoder_binding.bind_input("point_labels", &Tensor::from_array(point_labels)?)?;
        decoder_binding.bind_input("mask_input", &Tensor::from_array(mask_input)?)?;
        decoder_binding.bind_input("has_mask_input", &Tensor::from_array(array![0_f32])?)?;
        decoder_binding.bind_input(
            "orig_im_size",
            &Tensor::from_array(array![image_size.0, image_size.1])?,
        )?;

        let allocator = Allocator::new(
            decoder,
            MemoryInfo::new(
                AllocationDevice::CUDA_PINNED,
                RUNNING_SAM_DEVICE,
                AllocatorType::Device,
                MemoryType::CPUOutput,
            )?,
        )?;
        decoder_binding.bind_output_to_device("masks", &allocator.memory_info())?;
        decoder_binding.bind_output_to_device("iou_predictions", &allocator.memory_info())?;

        Ok(decoder.run_binding(decoder_binding)?)
    }
}

fn compute_launch_config(
    out_width: u32,
    out_height: u32,
    threads_per_block: (u32, u32, u32),
) -> LaunchConfig {
    // 计算网格尺寸
    let grid_dim_x = (out_width + threads_per_block.0 - 1) / threads_per_block.0;
    let grid_dim_y = (out_height + threads_per_block.1 - 1) / threads_per_block.1;
    let grid_dim_z = 1; // 对于 2D 图像处理，通常设置为 1

    LaunchConfig {
        grid_dim: (grid_dim_x, grid_dim_y, grid_dim_z),
        block_dim: threads_per_block,
        shared_mem_bytes: 0, // 假设不使用动态共享内存
    }
}
