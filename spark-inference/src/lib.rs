#![cfg_attr(debug_assertions, allow(warnings))]

pub mod engine;
pub mod utils;
pub mod inference;

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, LazyLock};
use cudarc::driver::CudaDevice;
use cudarc::nvrtc::compile_ptx;
use crate::utils::cuda::Cuda;

pub(crate) static IS_INIT: LazyLock<AtomicBool> = LazyLock::new(|| AtomicBool::new(false));
/*pub(crate) static CUDA_DEVICE: LazyLock<Arc<CudaDevice>> = LazyLock::new(|| {
    let device = CudaDevice::new(0).expect("Failed to open cuda device");

    let source_code = compile_ptx(r#"
            extern "C" __global__ void normalise_pixel(int *inp, const size_t numel) {
                unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < numel) {
                    inp[i] = static_cast<float>(inp[i]) / 255.0f;
                }
            }
        "#).expect("Failed to compile cuda function");

    let function_name = [
        "normalise_pixel"
    ];

    device.load_ptx(source_code, "inference", &function_name).expect("Failed to load cuda function");

    device
});*/
pub(crate) static INFERENCE_CUDA: LazyLock<Cuda> = LazyLock::new(|| Cuda::new(0).unwrap());


