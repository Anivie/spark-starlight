use std::ops::Deref;
use anyhow::Result;
use ort::Session;
use std::path::Path;
use std::sync::atomic::Ordering;
use crate::IS_INIT;

pub struct OnnxSession {
    pub(crate) session: Session
}

impl Deref for OnnxSession {
    type Target = Session;

    fn deref(&self) -> &Self::Target {
        &self.session
    }
}

impl OnnxSession {
    pub fn new(url: impl AsRef<Path>) -> Result<Self> {
        if !IS_INIT.load(Ordering::Relaxed) {
            // let provider = ort::TensorRTExecutionProvider::default().build().error_on_failure();
            let provider = ort::CUDAExecutionProvider::default().build().error_on_failure();
            ort::init()
                .with_execution_providers([provider])
                .commit()?;
            IS_INIT.store(true, Ordering::Relaxed);
        }

        let session = Session::builder()?
            .with_intra_threads(6)?
            .commit_from_file(url)?;

        Ok(OnnxSession { session })
    }
}