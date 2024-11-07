macro_rules! new_cuda {
    (
        [$model_name: literal],
        $(
            $name: ident => $source_code: literal,
        )*
    ) => {
        use std::collections::HashMap;
        use std::hash::Hash;
        use std::ops::Deref;
        use std::sync::Arc;
        use anyhow::{anyhow, Result};
        use cudarc::driver::{CudaDevice, CudaFunction};
        use cudarc::nvrtc::compile_ptx;

        pub(crate) struct Cuda {
            device: Arc<CudaDevice>,

            $(
                $name: CudaFunction,
            )*
        }

        impl Deref for Cuda {
            type Target = Arc<CudaDevice>;

            fn deref(&self) -> &Self::Target {
                &self.device
            }
        }

        impl Cuda {
            pub fn new(device_number: usize) -> Result<Self> {
                let device = CudaDevice::new(device_number)?;

                $(
                    device.load_ptx(
                        compile_ptx($source_code)?,
                        $model_name,
                        &[stringify!($name)],
                    )?;
                )*

                Ok(Self {
                    $(
                        $name: device
                            .get_func($model_name, stringify!($name))
                            .ok_or(anyhow!("Could not get function: {} from model: {}", stringify!($name), $model_name))?,
                    )*
                    device,
                })
            }

            $(
                pub fn $name(&self) -> CudaFunction {
                    self.$name.clone()
                }
            )*
        }
    };
}