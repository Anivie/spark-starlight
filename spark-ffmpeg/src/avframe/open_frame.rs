use crate::avframe::AVFrame;
use crate::ffi::{av_frame_alloc, av_image_fill_arrays, AVPixelFormat};
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

    pub fn fill_arrays(&mut self, data: *const u8, format: AVPixelFormat, width: i32, height: i32) -> Result<i32> {
        let need_size = native! {
            av_image_fill_arrays(
                self.data.as_ptr().cast_mut(),
                self.linesize.as_ptr().cast_mut(),
                data,
                format,
                width,
                height,
                1
            )
        };

        Ok(need_size)
    }
}