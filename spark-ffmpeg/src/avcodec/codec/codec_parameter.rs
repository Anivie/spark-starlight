use crate::avcodec::AVCodecContext;
use crate::ffi::{avcodec_parameters_alloc, avcodec_parameters_from_context};
use anyhow::{bail, Result};
use spark_proc_macro::wrap_ffmpeg;

wrap_ffmpeg!(
    AVCodecParameters drop+ [avcodec_parameters_free]
);

impl AVCodecParameters {
    pub fn new() -> Result<Self> {
        let ptr = unsafe { avcodec_parameters_alloc() };

        if ptr.is_null() {
            bail!("Failed to allocate AVCodecParameters.");
        }

        Ok(AVCodecParameters { inner: ptr })
    }

    pub(crate) fn from_context(avcodec_context: &AVCodecContext) -> Result<Self> {
        let parameters = AVCodecParameters::new()?;
        ffmpeg! {
            avcodec_parameters_from_context(parameters.inner, avcodec_context.inner)
        }
        Ok(parameters)
    }
}
