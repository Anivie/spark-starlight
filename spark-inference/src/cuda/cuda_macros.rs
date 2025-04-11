macro_rules! new_cuda {
    (
        $module: ident,
        $(
            $name: ident => $source_code: literal,
        )*
    ) => {
        use std::ops::Deref;
        use std::sync::Arc;
        use anyhow::{anyhow, Result};
        use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
        use cudarc::nvrtc::compile_ptx;

        pub(crate) struct $module {
            context: Arc<CudaContext>,
            module: Arc<CudaModule>,

            $(
                $name: CudaFunction,
            )*
        }

        impl Deref for $module {
            type Target = Arc<CudaContext>;

            fn deref(&self) -> &Self::Target {
                &self.context
            }
        }

        impl $module {
            pub fn new(device_number: usize) -> Result<Self> {
                let context = CudaContext::new(device_number)?;
                let module = context.load_module(compile_ptx(concat!($($source_code, )*))?)?;

                Ok(Self {
                    $(
                        $name: module.load_function(stringify!($name))?,
                    )*
                    context,
                    module,
                })
            }

            $(
                pub fn $name(&self) -> &CudaFunction {
                    &self.$name
                }
            )*
        }
    };
}
