use crate::engine::entity::box_point::Box;
use crate::engine::inference_engine::{ExecutionProvider, OnnxSession};
use anyhow::Result;
use bitvec::prelude::*;
use cudarc::driver::{DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig};
use log::debug;
use ndarray::{s, Axis, Ix2, Ix3};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::path::Path;
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::value::TensorRefMut;
use spark_media::filter::filter::AVFilter;
use spark_media::Image;
use crate::inference::{linear_interpolate, sigmoid};
use crate::INFERENCE_CUDA;

pub trait YoloDetectInference {
    fn inference_yolo(&self, tensor: Image, confidence: f32) -> Result<Vec<YoloDetectResult>>;
}

#[derive(Debug, Clone)]
pub struct YoloDetectResult {
    pub score: (usize, f32),

    pub x : f32,
    pub y : f32,
    pub width : f32,
    pub height : f32,
}

pub struct YoloDetectSession(OnnxSession);

impl YoloDetectSession {
    pub fn new(folder_path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self(
            OnnxSession::new(folder_path.as_ref().join("yolo_detect.onnx"), ExecutionProvider::CUDA)?
        ))
    }
}

impl YoloDetectInference for YoloDetectSession {
    fn inference_yolo(&self, mut image: Image, confidence: f32) -> Result<Vec<YoloDetectResult>> {
        let filter = AVFilter::builder(image.pixel_format()?, image.get_size())?
            .add_context("scale", "640:640:force_original_aspect_ratio=decrease")?
            .add_context("pad", "640:640:(ow-iw)/2:(oh-ih)/2:#727272")?
            .add_context("format", "rgb24")?
            .build()?;
        image.apply_filter(&filter)?;

        let tensor = {
            let buffer = INFERENCE_CUDA.htod_sync_copy(image.raw_data()?.as_slice())?;
            let cfg = LaunchConfig::for_num_elems((buffer.len() / 3) as u32);

            let tensor: TensorRefMut<'_, f32> = unsafe {
                let mut tensor = INFERENCE_CUDA.alloc::<f32>(buffer.len())?;

                INFERENCE_CUDA
                    .normalise_pixel_div()
                    .launch(cfg, (
                        &mut tensor,
                        &buffer,
                        buffer.len()
                    ))?;

                let back = TensorRefMut::from_raw(
                    MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default)?,
                    (*tensor.device_ptr() as usize as *mut ()).cast(),
                    vec![1, 3, 640, 640],
                )?;

                back
            };

            INFERENCE_CUDA.synchronize()?;

            tensor
        };

        debug!("Finish copying tensor to device");
        let outputs = self.0.run([tensor.into()])?;
        debug!("Finish running model");

        let output_first = outputs["output0"].try_extract_tensor::<f32>()?.t().into_owned();
        let output_first = output_first.squeeze().into_dimensionality::<Ix2>()?;

        let result = output_first
            .axis_iter(Axis(0))
            .into_par_iter()
            .filter(|box_output| {
                box_output
                    .slice(s![4 .. box_output.len()])
                    .iter()
                    .any(|&score| score > confidence)
            })
            .map(|box_output| {
                let score = box_output.slice(s![4 .. box_output.len()]);
                let (max_score, index) = {
                    let mut max_score = 0.0;
                    let mut index = 0;
                    for (i, &s) in score.iter().enumerate() {
                        if s > max_score {
                            max_score = s;
                            index = i;
                        }
                    }
                    (max_score, index)
                };

                (box_output, max_score, index)
            })
            .map(|(box_output, max_score, index)| {
                YoloDetectResult {
                    score: (index, max_score),
                    x: box_output[0],
                    y: box_output[1],
                    width: box_output[2],
                    height: box_output[3],
                }
            })
            .collect::<Vec<_>>();

        Ok(result)
    }
}

fn calculate_iou(a: &YoloDetectResult, b: &YoloDetectResult) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.width).min(b.x + b.width);
    let y2 = (a.y + a.height).min(b.y + b.height);

    let intersection = ((x2 - x1).max(0.0)) * ((y2 - y1).max(0.0));
    let union = a.width * a.height + b.width * b.height - intersection;

    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

fn non_maximum_suppression(anchors: Vec<YoloDetectResult>, iou_threshold: f32) -> Vec<YoloDetectResult> {
    let mut anchors = anchors;
    anchors.sort_by(|a, b| b.score.1.partial_cmp(&a.score.1).unwrap()); // 按分数降序排列

    let mut result = Vec::new();

    while !anchors.is_empty() {
        let current = anchors.remove(0); // 取出分数最高的锚点
        result.push(current.clone());

        // 保留与当前锚点 IoU 小于阈值的锚点
        anchors.retain(|anchor| calculate_iou(&current, anchor) < iou_threshold);
    }

    result
}