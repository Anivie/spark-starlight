use std::ffi::c_void;
use crate::avframe::{AVFrame, MarkAlloc};
use crate::ffi::{av_frame_alloc, av_image_alloc, av_image_fill_arrays, AVPixelFormat};
use anyhow::{anyhow, Result};

impl AVFrame {
    pub fn new() -> Result<Self> {
        let frame = unsafe {
            av_frame_alloc()
        };

        if frame.is_null() {
            Err(anyhow!("Failed to allocate frame"))
        } else {
            Ok(AVFrame { inner: frame, is_alloc_image: None })
        }
    }

    pub fn alloc_image(&mut self, format: AVPixelFormat, width: i32, height: i32) -> Result<i32> {
        let need_size = unsafe {
            av_image_alloc(
                self.data.as_ptr().cast_mut(),
                self.linesize.as_ptr().cast_mut(),
                width,
                height,
                format,
                1
            )
        };

        // self.is_alloc_image = unsafe {
        //     Some(MarkAlloc(*self.data.as_ptr().cast::<*mut c_void>()))
        // };

        Ok(need_size)
    }

    pub fn fill_arrays(&mut self, data: *const u8, format: AVPixelFormat, size: (i32, i32)) -> Result<i32> {
        let need_size = native! {
            av_image_fill_arrays(
                self.data.as_ptr().cast_mut(),
                self.linesize.as_ptr().cast_mut(),
                data,
                format,
                size.0,
                size.1,
                1
            )
        };

        Ok(need_size)
    }
}