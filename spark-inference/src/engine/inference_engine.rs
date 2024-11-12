use std::ops::Deref;
use anyhow::Result;
use ort::Session;
use std::path::Path;

pub struct OnnxSession {
    pub(crate) session: Session
}

pub enum ExecutionProvider {
    CUDA,
    TensorRT
}

impl Deref for OnnxSession {
    type Target = Session;

    fn deref(&self) -> &Self::Target {
        &self.session
    }
}

impl OnnxSession {
    pub fn new(url: impl AsRef<Path>, executor: ExecutionProvider) -> Result<Self> {
        let session = Session::builder()?
            .with_intra_threads(6)?
            .with_execution_providers(
                [
                    match executor {
                        ExecutionProvider::CUDA => ort::CUDAExecutionProvider::default().build().error_on_failure(),
                        ExecutionProvider::TensorRT => ort::TensorRTExecutionProvider::default().build().error_on_failure()
                    }
                ]
            )?
            .commit_from_file(url)?;

        Ok(OnnxSession { session })
    }
}