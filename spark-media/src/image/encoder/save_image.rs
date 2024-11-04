use crate::image::util::image_inner::ImageInner;
use crate::Image;
use anyhow::Result;
use spark_ffmpeg::avcodec::{AVCodec, AVCodecContext};
use spark_ffmpeg::avframe::AVFrame;
use spark_ffmpeg::avstream::AVCodecID;
use spark_ffmpeg::pixformat::AVPixelFormat;
use std::path::Path;

impl Image {
    pub fn new_with_empty(size: (i32, i32), pixel_format: AVPixelFormat, codec_id: AVCodecID) -> Result<Self> {
        let codec = AVCodec::new_encoder_with_id(codec_id)?;

        let mut codec_context = AVCodecContext::new(size, pixel_format, &codec, None)?;
        codec_context.set_size(size);
        codec_context.set_pixel_format(pixel_format);

        let frame = {
            let mut frame = AVFrame::new()?;
            frame.set_size(size);
            frame.alloc_image(pixel_format)?;
            frame
        };

        let inner = ImageInner {
            packet: None,
            frame,
        };

        Ok(Image {
            decoder: None,
            encoder: Some(codec_context),
            utils: Default::default(),
            inner
        })
    }

    pub fn save(&mut self, path: impl AsRef<Path>) -> Result<()> {
        self.try_encoder()?;

        let encoder = self.encoder.as_mut().unwrap();
        encoder.send_frame(&self.inner.frame)?;

        let packet = encoder.receive_packet()?;
        packet.save(path)?;

        self.inner.packet = Some(packet);

        Ok(())
    }
}