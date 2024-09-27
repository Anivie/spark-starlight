use std::ptr::copy;
use crate::avframe::AVFrame;
use crate::ffi::{av_frame_alloc, av_image_alloc, av_image_fill_arrays};
use crate::pixformat::AVPixelFormat;
use anyhow::{anyhow, Result};
use crate::DeepClone;

impl AVFrame {
    pub fn new() -> Result<Self> {
        let frame = unsafe {
            av_frame_alloc()
        };

        if frame.is_null() {
            Err(anyhow!("Failed to allocate frame"))
        } else {
            Ok(AVFrame { inner: frame })
        }
    }

    pub fn set_size(&mut self, size: (i32, i32)) {
        self.width = size.0;
        self.height = size.1;
    }

    pub fn set_format(&mut self, format: AVPixelFormat) {
        self.format = format as i32;
    }

    pub fn alloc_image(&mut self, format: AVPixelFormat, width: i32, height: i32) -> Result<i32> {
        let need_size = unsafe {
            av_image_alloc(
                self.data.as_ptr().cast_mut(),
                self.linesize.as_ptr().cast_mut(),
                width,
                height,
                format as i32,
                32
            )
        };

        Ok(need_size)
    }

    pub fn fill_arrays(&mut self, data: *const u8, format: AVPixelFormat, size: (i32, i32)) -> Result<i32> {
        let need_size = native! {
            av_image_fill_arrays(
                self.data.as_ptr().cast_mut(),
                self.linesize.as_ptr().cast_mut(),
                data,
                format as i32,
                size.0,
                size.1,
                1
            )
        };

        Ok(need_size)
    }
}

impl DeepClone for AVFrame {
    fn deep_clone(&self) -> Result<Self> {
        let mut new = AVFrame::new()?;
        let (new_ref, old_ref) = unsafe {
            (&mut *new.inner, &*self.inner)
        };

        new_ref.width = old_ref.width;
        new_ref.height = old_ref.height;
        new_ref.linesize = old_ref.linesize;
        new_ref.format = old_ref.format;
        new_ref.sample_rate = old_ref.sample_rate;
        new_ref.nb_samples = old_ref.nb_samples;
        new.alloc_image(AVPixelFormat::try_from(self.format)?, self.width, self.height)?;

        unsafe {
            copy(old_ref.data[0], new_ref.data[0], old_ref.linesize[0] as usize * old_ref.height as usize);
        }

        Ok(new)
    }
}