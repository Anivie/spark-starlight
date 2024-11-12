#![allow(dead_code)]
#![cfg_attr(debug_assertions, allow(warnings))]

extern crate openblas_src;

pub mod engine;
pub mod utils;
pub mod inference;
pub(crate) mod cuda;

use cuda::cuda::Cuda;
use std::sync::LazyLock;

pub(crate) static INFERENCE_CUDA: LazyLock<Cuda> = LazyLock::new(|| Cuda::new(0).unwrap());