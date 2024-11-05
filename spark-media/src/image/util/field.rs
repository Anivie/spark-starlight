use crate::Image;
use anyhow::{anyhow, Result};
use spark_ffmpeg::avcodec::{AVCodec, AVCodecContext};
use spark_ffmpeg::avstream::AVCodecID;
use spark_ffmpeg::pixformat::AVPixelFormat;
use std::mem::ManuallyDrop;

impl Image {
    pub(crate) fn available_codec(&self) -> &AVCodecContext {
        self.encoder.as_ref()
            .unwrap_or_else(|| self.decoder.as_ref().expect("At least one codec must be available"))
    }

    pub(crate) fn try_encoder(&mut self, codec_id: Option<u32>) -> Result<&AVCodecContext> {
        if self.encoder.is_none() {
            let encoder = AVCodec::new_encoder_with_id(
                codec_id
                    .unwrap_or_else(
                        || self.decoder.as_ref().ok_or(anyhow!("Init encoder need decoder exist.")).unwrap().id()
                    )
            )?;

            let context = AVCodecContext::from_frame(&encoder, &self.inner.frame, None)?;
            self.encoder = Some(context);
        }

        Ok(self.encoder.as_ref().unwrap())
    }

    pub(crate) fn try_decoder(&mut self, codec_id: Option<u32>) -> Result<&AVCodecContext> {
        if self.decoder.is_none() {
            let decoder = AVCodec::new_decoder_with_id(
                codec_id
                    .unwrap_or_else(
                        || self.encoder.as_ref().ok_or(anyhow!("Init decoder need encoder exist.")).unwrap().id()
                    )
            )?;

            let context = AVCodecContext::from_frame(&decoder, &self.inner.frame, None)?;
            self.decoder = Some(context);
        }

        Ok(self.decoder.as_ref().unwrap())
    }

    pub fn pixel_format(&self) -> Result<AVPixelFormat> {
        self.inner.frame.pixel_format()
    }

    pub fn codec_id(&self) -> AVCodecID {
        self.available_codec().id()
    }

    pub fn raw_data(&self) -> Result<ManuallyDrop<Vec<u8>>> {
        Ok(self.inner.frame.get_raw_data(0))
    }
}