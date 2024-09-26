use crate::engine::IS_INIT;
use anyhow::Result;
use ort::{CUDAExecutionProvider, Session};
use std::path::Path;
use std::sync::atomic::Ordering;

pub struct InferenceEngine {
    pub(crate) session: Session
}

impl InferenceEngine {
    pub fn new(url: impl AsRef<Path>) -> Result<Self> {
        if !IS_INIT.load(Ordering::Relaxed) {
            // let provider = TensorRTExecutionProvider::default().build().error_on_failure();
            let provider = CUDAExecutionProvider::default().build().error_on_failure();
            ort::init()
                .with_execution_providers([provider])
                .commit()?;
            IS_INIT.store(true, Ordering::Relaxed);
        }

        let session = Session::builder()?.commit_from_file(url)?;

        Ok(InferenceEngine { session })
    }
}