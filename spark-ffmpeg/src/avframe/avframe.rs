use crate::avframe::AVFrame;
use crate::ffi::{
    av_frame_alloc, av_frame_copy, av_frame_copy_props, av_frame_get_buffer, av_image_alloc,
    av_image_fill_arrays,
};
use crate::ffi_enum::AVPixelFormat;
use crate::DeepClone;
use anyhow::{anyhow, Result};

impl AVFrame {
    pub fn new() -> Result<Self> {
        let frame = unsafe { av_frame_alloc() };

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

    pub fn alloc_image(&mut self, format: AVPixelFormat) -> Result<i32> {
        let need_size = unsafe {
            av_image_alloc(
                self.data.as_ptr().cast_mut(),
                self.linesize.as_ptr().cast_mut(),
                self.width,
                self.height,
                format as i32,
                32,
            )
        };

        Ok(need_size)
    }

    pub fn pixel_format(&self) -> Result<AVPixelFormat> {
        Ok(AVPixelFormat::try_from(self.format)?)
    }

    pub fn fill_arrays(
        &mut self,
        data: *const u8,
        format: AVPixelFormat,
        size: (i32, i32),
    ) -> Result<i32> {
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
        let new = AVFrame::new()?;
        let (new_ref, old_ref) = unsafe { (&mut *new.inner, &*self.inner) };

        new_ref.format = old_ref.format;
        new_ref.width = old_ref.width;
        new_ref.height = old_ref.height;
        new_ref.nb_samples = old_ref.nb_samples;

        ffmpeg! {
            av_frame_get_buffer(new_ref as *mut crate::ffi::AVFrame, 32)
        }
        ffmpeg! {
            av_frame_copy(
                new_ref as *mut crate::ffi::AVFrame,
                old_ref as *const crate::ffi::AVFrame
            )
        }
        ffmpeg! {
            av_frame_copy_props(
                new_ref as *mut crate::ffi::AVFrame,
                old_ref as *const crate::ffi::AVFrame
            )
        }

        Ok(new)
    }
}
