use crate::Image;
use anyhow::{anyhow, Result};
use spark_ffmpeg::avcodec::{AVCodec, AVCodecContext};
use spark_ffmpeg::avframe::AVFrame;
use spark_ffmpeg::sws::SwsContext;
use std::ops::Deref;

pub trait ResizeImage {
    fn resize_to(&mut self, size: (i32, i32)) -> Result<()>;
    fn resize_into(&self, size: (i32, i32)) -> Result<Self>
    where
        Self: Sized;
}

impl ResizeImage for Image {
    fn resize_to(&mut self, size: (i32, i32)) -> Result<()> {
        let format = self.utils.format.as_ref().ok_or(anyhow!("Failed to get format."))?;
        let pixel_format = format.pixel_format(0)?;
        let sws = match &self.utils.sws {
            Some(sws) => sws,
            None => {
                let sws = SwsContext::from_format_context(&self.available_codec(), Some(pixel_format), Some((size.0, size.1)), None)?;
                self.utils.sws = Some(sws);
                self.utils.sws.as_ref().unwrap()
            }
        };

        let scaled_frame = {
            let mut scaled_frame = AVFrame::new()?;
            scaled_frame.set_size(size);
            scaled_frame.set_format(pixel_format);
            scaled_frame.alloc_image(pixel_format, size.0, size.1)?;
            scaled_frame
        };

        sws.scale_image(self.frame()?.deref(), &scaled_frame)?;

        let encoder = AVCodec::new_encoder_with_id(self.decoder.as_ref().unwrap().id())?;
        let context = AVCodecContext::from_frame(&encoder, &scaled_frame, None)?;

        context.send_frame(&scaled_frame)?;
        let packet = context.receive_packet()?;
        self.inner.replace_packet(packet);
        self.encoder = Some(context);

        Ok(())
    }

    fn resize_into(&self, size: (i32, i32)) -> Result<Self> {
        let format = self.utils.format.as_ref().ok_or(anyhow!("Failed to get format."))?;
        let pixel_format = format.pixel_format(0)?;

        let mut sws_temp = None;
        let sws = match &self.utils.sws {
            Some(sws) => sws,
            None => {
                sws_temp.replace(SwsContext::from_format_context(&self.available_codec(), Some(pixel_format), Some((size.0, size.1)), None)?);
                sws_temp.as_ref().unwrap()
            }
        };

        let scaled_frame = {
            let mut scaled_frame = AVFrame::new()?;
            scaled_frame.set_size(size);
            scaled_frame.alloc_image(pixel_format, size.0, size.1)?;
            scaled_frame
        };

        sws.scale_image(self.frame()?.deref(), &scaled_frame)?;

        let mut new = self.clone();
        if new.utils.sws.is_none() {
            new.utils.sws = sws_temp;
        }
        // new.available_codec().replace_frame(scaled_frame);

        Ok(new)
    }
}

impl Image {
    pub fn get_width(&self) -> i32 {
        self.available_codec().size().0
    }

    pub fn get_height(&self) -> i32 {
        self.available_codec().size().1
    }

    pub fn get_size(&self) -> (i32, i32) {
        self.available_codec().size()
    }
}