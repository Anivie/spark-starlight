use anyhow::Result;
use ort::session::Session;
use std::ops::{Deref, DerefMut};
use std::path::Path;

pub struct OnnxSession {
    pub(crate) session: Session,
    pub(crate) executor: ExecutionProvider,
}

#[derive(Copy, Clone, Debug)]
pub enum ExecutionProvider {
    CPU,
    CUDA(i32),
    TensorRT(i32),
}

impl Deref for OnnxSession {
    type Target = Session;

    fn deref(&self) -> &Self::Target {
        &self.session
    }
}

impl DerefMut for OnnxSession {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.session
    }
}

impl OnnxSession {
    pub fn new(url: impl AsRef<Path>, executor: ExecutionProvider) -> Result<Self> {
        let session = Session::builder()?
            .with_intra_threads(6)?
            .with_execution_providers([match executor {
                ExecutionProvider::CUDA(id) => {
                    ort::execution_providers::CUDAExecutionProvider::default()
                        .with_device_id(id)
                        .build()
                        .error_on_failure()
                }
                ExecutionProvider::TensorRT(id) => {
                    ort::execution_providers::TensorRTExecutionProvider::default()
                        .with_device_id(id)
                        .build()
                        .error_on_failure()
                }
                ExecutionProvider::CPU => ort::execution_providers::CPUExecutionProvider::default()
                    .build()
                    .error_on_failure(),
            }])?
            .commit_from_file(url)?;

        Ok(OnnxSession { session, executor })
    }
}
