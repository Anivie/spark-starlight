use crate::image::util::image_inner::ImageInner;
use crate::Image;
use anyhow::{anyhow, bail, Result};
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
        self.try_encoder(None)?;

        let encoder = self.encoder.as_mut().unwrap();
        encoder.send_frame(&self.inner.frame)?;

        let packet = encoder.receive_packet()?;
        packet.save(path)?;

        self.inner.packet = Some(packet);

        Ok(())
    }

    pub fn save_with_format(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();

        let extension = path.extension();
        if let Some(extension) = extension {
            let extension = extension.to_ascii_uppercase();
            let extension = extension.to_str().ok_or(anyhow!("Fail to cast extension to str."))?;
            let id = match extension {
                "PNG" => 61,
                _ => bail!("UNKNOWN FILE FORMAT")
            };
            self.try_encoder(Some(id))?;
        } else {
            self.try_encoder(None)?;
        }

        let encoder = self.encoder.as_mut().unwrap();
        encoder.send_frame(&self.inner.frame)?;

        let packet = encoder.receive_packet()?;
        packet.save(path)?;

        self.inner.packet = Some(packet);

        Ok(())
    }
}