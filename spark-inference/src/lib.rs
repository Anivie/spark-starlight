#![cfg_attr(debug_assertions, allow(warnings))]

pub mod engine;
pub mod utils;
pub mod inference;
pub(crate) mod cuda;

use cuda::cuda::Cuda;
use std::sync::LazyLock;

pub(crate) static INFERENCE_CUDA: LazyLock<Cuda> = LazyLock::new(|| Cuda::new(0).unwrap());

pub fn init_inference_engine() -> anyhow::Result<()> {
    let provider = ort::CUDAExecutionProvider::default().build().error_on_failure();
    ort::init()
        .with_execution_providers([provider])
        .commit()?;

    Ok(())
}
