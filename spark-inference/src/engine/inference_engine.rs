use std::ops::Deref;
use anyhow::Result;
use ort::Session;
use std::path::Path;

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
        let session = Session::builder()?
            .with_intra_threads(6)?
            .commit_from_file(url)?;

        Ok(OnnxSession { session })
    }
}