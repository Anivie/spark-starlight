use std::mem::forget;
use crate::Image;
use anyhow::{anyhow, Result};
use rayon::prelude::*;
use spark_ffmpeg::avframe::AVFrame;
use spark_ffmpeg::sws::SwsContext;
use spark_ffmpeg::CloneFrom;
use std::ptr::{copy, copy_nonoverlapping};
use spark_ffmpeg::util::ptr_wrapper::SafePtr;

pub trait ResizeImage {
    fn resize_to(&mut self, size: (i32, i32)) -> Result<()>;
    fn resize_into(&self, size: (i32, i32)) -> Result<Self>
    where
        Self: Sized;
}

impl Image {
    pub fn resize(&mut self) -> Result<()> {
        let max = self.get_width().max(self.get_height());
        println!("{}", self.inner.frame.line_size(0));

        let mut frame = AVFrame::new()?;
        frame.clone_copy_fields(&self.inner.frame);
        frame.set_size((max, max));
        frame.alloc_image(self.pixel_format())?;
        frame.get_raw_data(0).par_iter_mut().for_each(|x| *x = 0);

        let origin_data = self.inner.frame.get_raw_data(0);

        //把图片填充在中间
        if self.get_width() > self.get_height() {
            let need = max - self.get_height();

            let ptr = unsafe {
                frame
                    .get_raw_data(0)
                    .as_mut_ptr()
                    .add(((need / 2) * self.inner.frame.line_size(0)) as usize)
            };
            let ptr = SafePtr::new(ptr);

            (0..(self.get_height() * self.inner.frame.line_size(0))as usize)
                .into_iter()
                .for_each(|index| {
                    unsafe {
                        ptr.add(index).write(origin_data[index]);
                    }
                });
        }
        self.inner.frame = frame;


        Ok(())
    }
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
            scaled_frame.alloc_image(pixel_format)?;
            scaled_frame
        };

        self.encoder
            .as_mut()
            .map(|x| {
                x.set_size(size);
            });

        self.decoder
            .as_mut()
            .map(|x| {
                x.set_size(size);
            });

        sws.scale_image(&self.inner.frame, &scaled_frame)?;
        self.inner.frame = scaled_frame;

        Ok(())
    }

    fn resize_into(&self, size: (i32, i32)) -> Result<Self> {
        let mut new_image = self.clone();
        let format = new_image.utils.format.as_ref().ok_or(anyhow!("Failed to get format."))?;
        let pixel_format = format.pixel_format(0)?;
        let sws = match &new_image.utils.sws {
            Some(sws) => sws,
            None => {
                let sws = SwsContext::from_format_context(&new_image.available_codec(), Some(pixel_format), Some((size.0, size.1)), None)?;
                new_image.utils.sws = Some(sws);
                new_image.utils.sws.as_ref().unwrap()
            }
        };

        let scaled_frame = {
            let mut scaled_frame = AVFrame::new()?;
            scaled_frame.set_size(size);
            scaled_frame.set_format(pixel_format);
            scaled_frame.alloc_image(pixel_format)?;
            scaled_frame
        };

        new_image.encoder
            .as_mut()
            .map(|x| {
                x.set_size(size);
            });

        new_image.decoder
            .as_mut()
            .map(|x| {
                x.set_size(size);
            });

        sws.scale_image(&new_image.inner.frame, &scaled_frame)?;
        new_image.inner.frame = scaled_frame;

        Ok(new_image)
    }
}

impl Image {
    pub fn get_width(&self) -> i32 {
        self.inner.frame.get_width()
    }

    pub fn get_height(&self) -> i32 {
        self.inner.frame.get_height()
    }

    pub fn get_size(&self) -> (i32, i32) {
        (self.get_width(), self.get_height())
    }
}