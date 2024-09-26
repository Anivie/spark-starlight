use crate::avcodec::AVCodecContext;
use crate::ffi::{avcodec_parameters_to_context, AVCodecParameters, AVStream};

impl AVCodecContext {
    pub(super) fn apply_format(&mut self, frame: &AVStream) -> anyhow::Result<()> {
        ffmpeg! {
            avcodec_parameters_to_context(self.inner, (*frame).codecpar as *const AVCodecParameters)
        }

        Ok(())
    }

    pub(super) fn apply_format_with_parameter(&mut self, format_context: &AVCodecParameters) -> anyhow::Result<()> {
        ffmpeg! {
            avcodec_parameters_to_context(self.inner, format_context as *const AVCodecParameters)
        }

        Ok(())
    }

    pub fn resize(&mut self, size: (i32, i32)) {
        self.width = size.0;
        self.height = size.1;
    }
}