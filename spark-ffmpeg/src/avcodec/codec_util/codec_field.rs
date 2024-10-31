use crate::avcodec::AVCodecContext;
use crate::ffi::av_image_get_buffer_size;
use crate::pixformat::AVPixelFormat;
use anyhow::anyhow;

impl AVCodecContext {
    pub fn size(&self) -> (i32, i32) {
        let context = unsafe { *self.inner };
        (context.width, context.height)
    }

    pub fn set_size(&mut self, size: (i32, i32)) {
        self.width = size.0;
        self.height = size.1;
    }

    pub fn buffer_size(&self, format: AVPixelFormat) -> anyhow::Result<i32> {
        let size = unsafe {
            av_image_get_buffer_size(
                format as i32,
                self.width,
                self.height,
                1,
            )
        };

        if size > 0 {
            Ok(size)
        } else {
            Err(anyhow!("Failed to get buffer size with code {}.", size))
        }
    }

    pub fn id(&self) -> u32 {
        self.codec_id
    }

    pub fn pixel_format(&self) -> AVPixelFormat {
        AVPixelFormat::try_from(self.pix_fmt).unwrap()
    }

    pub fn set_pixel_format(&mut self, format: AVPixelFormat) {
        self.pix_fmt = format as i32;
    }
}