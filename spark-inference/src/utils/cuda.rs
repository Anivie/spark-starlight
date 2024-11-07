/*use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Deref;
use std::sync::Arc;
use anyhow::{anyhow, Result};
use cudarc::driver::{CudaDevice, CudaFunction};
use cudarc::nvrtc::compile_ptx;

pub(crate) struct Cuda {
    device: Arc<CudaDevice>,

    function: HashMap<String, CudaFunction>
}

impl Deref for Cuda {
    type Target = Arc<CudaDevice>;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl Cuda {
    pub fn new() -> Result<Self> {
        let device = CudaDevice::new(0)?;

        let source_code = compile_ptx(r#"
            extern "C" __global__ void normalise_pixel(float *out, const int *inp, const size_t numel) {
                unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < numel) {
                    out[i] = static_cast<float>(inp[i]) / 255.0f;
                }
            }
        "#)?;

        let function_name = [
            "normalise_pixel"
        ];

        device.load_ptx(source_code, "inference", &function_name)?;

        let function = vec![
            device
                .get_func("inference", "normalise_pixel")
                .ok_or(anyhow!("Could not get function normalise_pixel"))?
        ];

        let function = function
            .into_iter()
            .zip(function_name.iter())
            .map(|(f, n)| (n.to_string(), f))
            .collect();

        Ok(Self {
            device,
            function
        })
    }

    pub fn get_func(&self, func: &str) -> Option<&CudaFunction> {
        self.function.get(func)
    }
}*/

new_cuda! {
    ["inference"],
    normalise_pixel => r#"
        extern "C" __global__ void normalise_pixel(float *out, const unsigned char *inp, const size_t numel) {
            unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < numel) {
                out[i] = static_cast<float>(inp[i]) / 255.0f;
            }
        }
    "#,
}