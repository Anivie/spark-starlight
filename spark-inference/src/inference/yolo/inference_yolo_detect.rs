use crate::engine::inference_engine::{ExecutionProvider, OnnxSession};
use crate::{INFERENCE_YOLO, RUNNING_YOLO_DEVICE};
use anyhow::Result;
use cudarc::driver::{DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig};
use log::debug;
use ndarray::{s, Axis, Ix2};
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::value::TensorRefMut;
use rayon::prelude::*;
use spark_media::filter::filter::AVFilter;
use spark_media::Image;
use std::path::Path;

pub trait YoloDetectInference {
    fn inference_yolo(&self, tensor: Image, confidence: f32) -> Result<Vec<YoloDetectResult>>;
}

#[derive(Debug, Clone)]
pub struct YoloDetectResult {
    pub score: Vec<f32>,

    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

pub struct YoloDetectSession(OnnxSession);

impl YoloDetectSession {
    pub fn new(folder_path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self(OnnxSession::new(
            folder_path.as_ref().join("yolo_detect.onnx"),
            ExecutionProvider::CUDA(RUNNING_YOLO_DEVICE),
        )?))
    }
}

impl YoloDetectInference for YoloDetectSession {
    fn inference_yolo(&self, mut image: Image, confidence: f32) -> Result<Vec<YoloDetectResult>> {
        let filter = AVFilter::builder(image.pixel_format()?, image.get_size())?
            .add_context("scale", "640:640:force_original_aspect_ratio=decrease")?
            .add_context("pad", "640:640:(ow-iw)/2:(oh-ih)/2:#727272")?
            .add_context("format", "rgb24")?
            .build()?;

        let (image_width, image_height) = image.get_size();
        let (image_width, image_height) = (image_width as f32, image_height as f32);

        image.apply_filter(&filter)?;

        let tensor = {
            let buffer = INFERENCE_YOLO.htod_sync_copy(image.raw_data()?.as_slice())?;
            let cfg = LaunchConfig::for_num_elems((buffer.len() / 3) as u32);

            let tensor: TensorRefMut<'_, f32> = unsafe {
                let mut tensor = INFERENCE_YOLO.alloc::<f32>(buffer.len())?;

                INFERENCE_YOLO
                    .normalise_pixel_div()
                    .launch(cfg, (&mut tensor, &buffer, buffer.len()))?;

                let back = TensorRefMut::from_raw(
                    MemoryInfo::new(
                        AllocationDevice::CUDA,
                        RUNNING_YOLO_DEVICE,
                        AllocatorType::Device,
                        MemoryType::Default,
                    )?,
                    (*tensor.device_ptr() as usize as *mut ()).cast(),
                    vec![1, 3, 640, 640],
                )?;

                back
            };

            tensor
        };

        debug!("Finish copying tensor to device");
        let outputs = self.0.run([tensor.into()])?;
        debug!("Finish running model");

        let output_first = outputs["output0"]
            .try_extract_tensor::<f32>()?
            .t()
            .into_owned();
        let output_first = output_first.squeeze().into_dimensionality::<Ix2>()?;

        let result = output_first
            .axis_iter(Axis(0))
            .into_par_iter()
            .filter(|box_output| {
                box_output
                    .slice(s![4..box_output.len()])
                    .iter()
                    .any(|&score| score > confidence)
            })
            .map(|box_output| {
                let score = box_output.slice(s![4..box_output.len()]).to_vec();
                let (x, y, width, height) = yolo_to_image_coords(
                    box_output[0],
                    box_output[1],
                    box_output[2],
                    box_output[3],
                    image_width,
                    image_height,
                    640.0,
                    640.0,
                );
                YoloDetectResult {
                    score,
                    x,
                    y,
                    width,
                    height,
                }
            })
            .collect::<Vec<_>>();

        Ok(result)
    }
}

fn yolo_to_image_coords(
    x_center: f32,
    y_center: f32,
    width: f32,
    height: f32,
    img_width: f32,
    img_height: f32,
    input_width: f32,
    input_height: f32,
) -> (f32, f32, f32, f32) {
    // 计算缩放比例
    let scale = f32::max(img_width / input_width, img_height / input_height);

    // 计算原图在等比例缩放后的尺寸
    let scaled_img_width = img_width / scale;
    let scaled_img_height = img_height / scale;

    // 计算平移的量
    let x_move = (scaled_img_width - input_width).abs() / 2.0;
    let y_move = (scaled_img_height - input_height).abs() / 2.0;

    // 映射回原图的坐标系
    let ret_x_center = (x_center - x_move) * scale;
    let ret_y_center = (y_center - y_move) * scale;
    let ret_width = width * scale;
    let ret_height = height * scale;

    (ret_x_center, ret_y_center, ret_width, ret_height)
}
