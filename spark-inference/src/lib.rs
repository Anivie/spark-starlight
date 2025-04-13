#![feature(let_chains)]
#![allow(dead_code)]
#![cfg_attr(debug_assertions, allow(warnings))]

pub(crate) mod cuda;
pub mod engine;
pub mod inference;
pub mod utils;

use crate::cuda::cuda_sam::CudaSam;
use crate::cuda::cuda_yolo::CudaYolo;
use std::sync::LazyLock;

const RUNNING_SAM_DEVICE: i32 = 0;
const RUNNING_YOLO_DEVICE: i32 = 0;

pub(crate) static INFERENCE_SAM: LazyLock<CudaSam> =
    LazyLock::new(|| CudaSam::new(RUNNING_SAM_DEVICE as usize).unwrap());
pub(crate) static INFERENCE_YOLO: LazyLock<CudaYolo> =
    LazyLock::new(|| CudaYolo::new(RUNNING_YOLO_DEVICE as usize).unwrap());

pub use spark_ffmpeg::disable_ffmpeg_logging;
