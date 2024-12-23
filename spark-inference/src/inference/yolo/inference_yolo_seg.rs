use crate::engine::entity::box_point::Box;
use crate::engine::inference_engine::OnnxSession;
use crate::utils::tensor::{linear_interpolate, sigmoid};
use crate::{INFERENCE_YOLO, RUNNING_YOLO_DEVICE};
use anyhow::Result;
use bitvec::prelude::*;
use cudarc::driver::{DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig};
use log::debug;
use ndarray::{s, Axis, Ix2, Ix3};
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::value::TensorRefMut;
use rayon::prelude::*;
use spark_media::filter::filter::AVFilter;
use spark_media::Image;
use std::cmp::Ordering;

pub trait YoloSegmentInference {
    fn inference_yolo(
        &self,
        tensor: Image,
        confidence: f32,
        probability_mask: f32,
    ) -> Result<Vec<YoloInferenceResult>>;
}

#[derive(Debug, Clone)]
pub struct YoloInferenceResult {
    pub boxed: Box,
    pub classify: usize,
    pub mask: BitVec,
    pub score: f32,
}

impl YoloSegmentInference for OnnxSession {
    fn inference_yolo(
        &self,
        mut image: Image,
        conf_thres: f32,
        iou_thres: f32,
    ) -> Result<Vec<YoloInferenceResult>> {
        let filter = AVFilter::builder(image.pixel_format()?, image.get_size())?
            .add_context("scale", "640:640:force_original_aspect_ratio=decrease")?
            .add_context("pad", "640:640:(ow-iw)/2:(oh-ih)/2:#727272")?
            .add_context("format", "rgb24")?
            .build()?;
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
        let outputs = self.session.run([tensor.into()])?;
        debug!("Finish running model");

        let output_first = outputs["output0"]
            .try_extract_tensor::<f32>()?
            .t()
            .into_owned();
        let output_first = output_first.squeeze().into_dimensionality::<Ix2>()?;

        let output_second = outputs["output1"].try_extract_tensor::<f32>()?.into_owned();
        let output_second = output_second.squeeze().into_dimensionality::<Ix3>()?;
        let output_second = output_second.to_shape((32, 25600))?;

        debug!("output_first: {:?}", output_first.shape());
        debug!("output_second: {:?}", output_second.shape());

        let mask = output_first
            .axis_iter(Axis(0))
            .into_par_iter()
            .filter(|box_output| {
                box_output
                    .slice(s![4..box_output.len() - 32])
                    .iter()
                    .any(|&score| score > conf_thres)
            })
            .map(|box_output| {
                let score = box_output.slice(s![4..box_output.len() - 32]);
                let score = score
                    .iter()
                    .max_by(|&a, &b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .unwrap_or(&0.0);
                (box_output, *score)
            })
            .map(|(box_output, score)| {
                let (x, y, w, h) = (box_output[0], box_output[1], box_output[2], box_output[3]);
                let x1 = x - w / 2.;
                let y1 = y - h / 2.;
                let x2 = x + w / 2.;
                let y2 = y + h / 2.;

                let feature_mask = box_output.slice(s![box_output.len() - 32..]);
                let feature_mask = feature_mask.to_shape((1, 32)).unwrap();

                let mask = {
                    let mask = feature_mask.dot(&output_second);
                    let mask = mask.into_shape_with_order((160, 160)).unwrap();
                    let mask = linear_interpolate(mask.into_owned(), (640, 640));
                    let mask = sigmoid(mask);
                    let x1 = x1 as usize;
                    let y1 = y1 as usize;
                    let x2 = x2 as usize;
                    let y2 = y2 as usize;

                    mask.iter()
                        .enumerate()
                        .map(|(index, value)| {
                            let y = index / 640;
                            let x = index % 640;
                            (y1..y2).contains(&y) && (x1..x2).contains(&x) && *value > iou_thres
                        })
                        .collect::<BitVec>()
                };

                let boxed = Box { x1, y1, x2, y2 };
                let classify = box_output
                    .slice(s![4..box_output.len() - 32])
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(index, _)| index)
                    .unwrap();

                YoloInferenceResult {
                    boxed,
                    classify,
                    mask,
                    score,
                }
            })
            .collect::<Vec<_>>();

        Ok(mask)
    }
}
