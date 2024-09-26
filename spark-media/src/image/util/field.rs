use crate::Image;
use anyhow::{anyhow, Result};
use parking_lot::{RwLockReadGuard, RwLockWriteGuard};
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

    /*pub(crate) fn try_encoder(&mut self) -> Result<&AVCodecContext> {
        let encoder = match self.encoder.as_ref() {
            Some(some) => some,
            None => {
                let decoder = self.decoder.as_ref().ok_or(anyhow!("Init encoder need decoder exist."))?;
                let encoder = AVCodec::new_encoder_with_id(decoder.id())?;
                let frame = self.inner.frame.as_ref().ok_or(anyhow!("Init encoder need frame exist."))?;

                let context = AVCodecContext::from_frame(&encoder, frame.read().deref(), None)?;

                self.encoder = Some(context);
                self.encoder.as_ref().unwrap()
            }
        };

        Ok(encoder)
    }

    pub(crate) fn try_decoder(&mut self) -> Result<&AVCodecContext> {
        let decoder = match self.decoder.as_ref() {
            Some(some) => some,
            None => {
                let encoder = self.encoder.as_ref().ok_or(anyhow!("Init decoder need encoder exist."))?;
                let decoder = AVCodec::new_decoder_with_id(encoder.id())?;
                let frame = self.inner.frame.as_ref().ok_or(anyhow!("Init encoder need frame exist."))?;

                let context = AVCodecContext::from_frame(&decoder, frame.read().deref(), None)?;
                self.decoder = Some(context);
                self.decoder.as_ref().unwrap()
            }
        };

        Ok(decoder)
    }*/

    pub fn frame(&self) -> Result<RwLockReadGuard<AVFrame>> {
        let frame = self.inner.frame.as_ref().ok_or(anyhow!("No frame are available"))?;
        Ok(frame.read())
    }

    pub fn frame_mut(&self) -> Result<RwLockWriteGuard<AVFrame>> {
        let frame = self.inner.frame.as_ref().ok_or(anyhow!("No frame are available"))?;
        Ok(frame.write())
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