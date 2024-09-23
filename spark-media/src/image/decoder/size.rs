use anyhow::Result;
use spark_ffmpeg::avframe::AVFrame;
use spark_ffmpeg::pixformat::AVPixelFormat;
use spark_ffmpeg::sws::SwsContext;
use crate::Image;

pub trait ResizeImage {
    fn resize(&mut self, size: (i32, i32), format: AVPixelFormat) -> Result<()>;
    fn resize_into(&self, size: (i32, i32), format: AVPixelFormat) -> Result<Self>
    where
        Self: Sized;
}

impl ResizeImage for Image {
    fn resize(&mut self, size: (i32, i32), format: AVPixelFormat) -> Result<()> {
        let sws = match &self.utils.sws {
            Some(sws) => sws,
            None => {
                let sws = SwsContext::from_format_context(&self.codec, Some(format), Some((size.0, size.1)), None)?;
                self.utils.sws = Some(sws);
                self.utils.sws.as_ref().unwrap()
            }
        };

        let scaled_frame = {
            let mut scaled_frame = AVFrame::new()?;
            scaled_frame.set_size(size.0, size.1);
            scaled_frame.alloc_image(format, size.0, size.1)?;
            scaled_frame
        };

        sws.scale_image(self.codec.last_frame(), &scaled_frame)?;
        self.codec.replace_frame(scaled_frame);

        Ok(())
    }

    fn resize_into(&self, size: (i32, i32), format: AVPixelFormat) -> Result<Self> {
        let mut sws_temp = None;
        let sws = match &self.utils.sws {
            Some(sws) => sws,
            None => {
                sws_temp.replace(SwsContext::from_format_context(&self.codec, Some(format), Some((size.0, size.1)), None)?);
                sws_temp.as_ref().unwrap()
            }
        };

        let scaled_frame = {
            let mut scaled_frame = AVFrame::new()?;
            scaled_frame.set_size(size.0, size.1);
            scaled_frame.alloc_image(format, size.0, size.1)?;
            scaled_frame
        };

        sws.scale_image(self.codec.last_frame(), &scaled_frame)?;

        let mut new = self.clone();
        if new.utils.sws.is_none() {
            new.utils.sws = sws_temp;
        }
        new.codec.replace_frame(scaled_frame);

        Ok(new)
    }
}

impl Image {
    pub fn get_width(&self) -> i32 {
        self.codec.size().0
    }

    pub fn get_height(&self) -> i32 {
        self.codec.size().1
    }

    pub fn get_size(&self) -> (i32, i32) {
        self.codec.size()
    }
}