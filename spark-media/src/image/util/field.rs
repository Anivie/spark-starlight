use crate::Image;
use anyhow::{anyhow, Result};
use parking_lot::RwLockReadGuard;
use spark_ffmpeg::avcodec::AVCodecContext;
use spark_ffmpeg::avframe::AVFrame;
use spark_ffmpeg::avstream::AVCodecID;
use spark_ffmpeg::pixformat::AVPixelFormat;
use std::mem::ManuallyDrop;

impl Image {
    pub(crate) fn available_codec(&self) -> &AVCodecContext {
        self.encoder.as_ref()
            .unwrap_or_else(|| self.decoder.as_ref().expect("At least one codec must be available"))
    }

    pub fn frame(&self) -> Result<RwLockReadGuard<AVFrame>> {
        let frame = self.inner.frame.as_ref().ok_or(anyhow!("No frame are available"))?;
        Ok(frame.read())
    }

    pub fn pixel_format(&self) -> AVPixelFormat {
        self.available_codec().pixel_format()
    }

    pub fn codec_id(&self) -> AVCodecID {
        self.available_codec().id()
    }

    pub fn raw_data(&self) -> Result<ManuallyDrop<Vec<u8>>> {
        Ok(self.frame()?.get_raw_data(0))
    }
}