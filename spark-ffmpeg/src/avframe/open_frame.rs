use crate::avframe::AVFrame;
use crate::ffi::{av_frame_alloc, av_image_alloc, av_image_fill_arrays};
use crate::pixformat::AVPixelFormat;
use anyhow::{anyhow, Result};

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

    pub fn set_size(&mut self, width: i32, height: i32) {
        self.width = width;
        self.height = height;
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
                1
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