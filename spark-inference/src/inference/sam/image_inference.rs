use crate::engine::inference_engine::{ExecutionProvider, OnnxSession};
use crate::utils::graph::SamPrompt;
use crate::{INFERENCE_SAM, RUNNING_SAM_DEVICE};
use anyhow::{anyhow, Result};
use bitvec::prelude::BitVec;
use cudarc::driver::{CudaSlice, DevicePtr, LaunchConfig, PushKernelArg};
use log::info;
use ndarray::{array, s, Array1, Array2, Array4, Axis};
use ort::inputs;
use ort::io_binding::IoBinding;
use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::session::SessionOutputs;
use ort::tensor::Shape;
use ort::value::{Tensor, TensorRefMut};
use parking_lot::Mutex;
use spark_media::filter::filter::AVFilter;
use spark_media::Image;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::sync::LazyLock;

pub trait SamImageInference {
    fn inference_frame(
        &self,
        image: Image,
        out_put_size: Option<(i32, i32)>,
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
        out_put_size: Option<(i32, i32)>,
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
                    out_put_size,
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

                let pred_mask = mask_decoder_output["masks"].try_extract_array::<f32>()?;

                let mut temp = BitVec::with_capacity(pred_mask.len());
                pred_mask
                    .slice(s![0, max_index, .., ..])
                    .iter()
                    .for_each(|x| {
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
            let stream = INFERENCE_SAM.new_stream().unwrap();
            stream.memcpy_stod(&[0.485, 0.456, 0.406]).unwrap()
        });
        static STD: LazyLock<CudaSlice<f32>> = LazyLock::new(|| {
            let stream = INFERENCE_SAM.new_stream().unwrap();
            stream.memcpy_stod(&[0.229, 0.224, 0.225]).unwrap()
        });
        let buffer = {
            let stream = INFERENCE_SAM.new_stream()?;
            stream.memcpy_stod(image.as_slice())?
        };
        let cfg = LaunchConfig::for_num_elems((image.len() / 3) as u32);

        let (stream, tensor) = unsafe {
            let image_size = image.len();
            let stream = INFERENCE_SAM.new_stream()?;
            let mut tensor = stream.alloc::<f32>(image_size)?;

            let mut builder = stream.launch_builder(INFERENCE_SAM.normalise_pixel_mean());
            builder.arg(&mut tensor);
            builder.arg(&buffer);
            builder.arg(MEAN.deref());
            builder.arg(STD.deref());
            builder.arg(&image_size);
            builder.launch(cfg)?;

            (stream, tensor)
        };

        let session_outputs = match image_encoder.executor {
            ExecutionProvider::CPU => {
                let tensor = stream.memcpy_dtov(&tensor)?;
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
                    (tensor.device_ptr(&stream).0 as usize as *mut ()).cast(),
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
        out_put_size: Option<(i32, i32)>,
        decoder_binding: &'a mut IoBinding,
        decoder: &'b mut OnnxSession,
    ) -> Result<SessionOutputs<'a, 'b>> {
        let mask_input = Array4::<f32>::zeros((1, 1, 256, 256));

        let trans_prompt = match &prompt {
            SamPrompt::Box(boxes) => {
                array![
                    1024f32 * ((boxes.x - boxes.width / 2_f32) / image_size.0 as f32),
                    1024f32 * ((boxes.y - boxes.height / 2_f32) / image_size.1 as f32),
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
            SamPrompt::Points(points) => {
                let mut base = Array1::zeros(points.len() * 2);
                for (i, point) in points.iter().enumerate() {
                    base[i * 2] = 1024f32 * (point.x / image_size.0 as f32);
                    base[i * 2 + 1] = 1024f32 * (point.y / image_size.1 as f32);
                }
                base
            }
            SamPrompt::Both(point, boxes) => {
                array![
                    1024f32 * (point.x / image_size.0 as f32),
                    1024f32 * (point.y / image_size.1 as f32),
                    1024f32 * ((boxes.x - boxes.width / 2_f32) / image_size.0 as f32),
                    1024f32 * ((boxes.y - boxes.height / 2_f32) / image_size.1 as f32),
                    1024f32 * ((boxes.x + boxes.width / 2_f32) / image_size.0 as f32),
                    1024f32 * ((boxes.y + boxes.height / 2_f32) / image_size.1 as f32),
                ]
            }
        };

        let point_labels = match prompt {
            SamPrompt::Point(_) => array![[1_f32]],
            SamPrompt::Points(point) => {
                let mut point_labels = Array2::zeros((1, point.len()));
                for (i, _) in point.iter().enumerate() {
                    point_labels[[0, i]] = 1_f32;
                }
                point_labels
            }
            SamPrompt::Box(_) => array![[2_f32, 3_f32]],
            SamPrompt::Both(_, _) => array![[1_f32, 2_f32, 3_f32]],
        };

        let prompt_out = {
            let size = trans_prompt.shape()[0] / 2;
            trans_prompt.into_shape_with_order((1, size, 2))?
        };

        decoder_binding.bind_input("point_coords", &Tensor::from_array(prompt_out)?)?;
        decoder_binding.bind_input("point_labels", &Tensor::from_array(point_labels)?)?;
        decoder_binding.bind_input("mask_input", &Tensor::from_array(mask_input)?)?;
        decoder_binding.bind_input("has_mask_input", &Tensor::from_array(array![0_f32])?)?;
        decoder_binding.bind_input(
            "orig_im_size",
            &Tensor::from_array(
                out_put_size
                    .map(|x| array![x.0, x.1])
                    .unwrap_or(array![image_size.0, image_size.1]),
            )?,
        )?;

        let allocator = Allocator::new(
            decoder,
            MemoryInfo::new(
                AllocationDevice::CPU,
                0,
                AllocatorType::Device,
                MemoryType::CPUOutput,
            )?,
        )?;
        decoder_binding.bind_output_to_device("masks", &allocator.memory_info())?;
        decoder_binding.bind_output_to_device("iou_predictions", &allocator.memory_info())?;

        Ok(decoder.run_binding(decoder_binding)?)
    }
}

pub fn compute_launch_config(out_width: u32, out_height: u32) -> LaunchConfig {
    let block_dim_x: u32 = 16;
    let block_dim_y: u32 = 16;
    let block_dim_z: u32 = 1;

    let grid_dim_x = (out_width + block_dim_x - 1) / block_dim_x;
    let grid_dim_y = (out_height + block_dim_y - 1) / block_dim_y;
    let grid_dim_z = 1;

    let shared_mem_bytes: u32 = 0;

    LaunchConfig {
        grid_dim: (grid_dim_x, grid_dim_y, grid_dim_z),
        block_dim: (block_dim_x, block_dim_y, block_dim_z),
        shared_mem_bytes,
    }
}
