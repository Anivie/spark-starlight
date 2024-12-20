#![feature(let_chains)]
#![allow(dead_code)]
#![cfg_attr(debug_assertions, allow(warnings))]

// extern crate openblas_src;

pub(crate) mod cuda;
pub mod engine;
pub mod inference;
pub mod utils;

use cuda::cuda::Cuda;
use std::sync::LazyLock;

const RUNNING_SAM_DEVICE: i32 = 0;
const RUNNING_YOLO_DEVICE: i32 = 0;

pub(crate) static INFERENCE_SAM: LazyLock<Cuda> =
    LazyLock::new(|| Cuda::new(RUNNING_SAM_DEVICE as usize).unwrap());
pub(crate) static INFERENCE_YOLO: LazyLock<Cuda> =
    LazyLock::new(|| Cuda::new(RUNNING_YOLO_DEVICE as usize).unwrap());
