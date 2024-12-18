#![allow(dead_code)]
#![cfg_attr(debug_assertions, allow(warnings))]

// extern crate openblas_src;

pub(crate) mod cuda;
pub mod engine;
pub mod inference;
pub mod utils;

use cuda::cuda::Cuda;
use std::sync::LazyLock;

pub(crate) static INFERENCE_CUDA: LazyLock<Cuda> = LazyLock::new(|| Cuda::new(0).unwrap());
