use crate::avframe::AVFrame;
use crate::pixformat::AVPixelFormat;
use rayon::iter::IntoParallelIterator;
use rayon::iter::*;
use std::ops::Deref;

struct SafeVecPtr(*mut u8);
impl Deref for SafeVecPtr {
    type Target = *mut u8;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
unsafe impl Send for SafeVecPtr {}
unsafe impl Sync for SafeVecPtr {}

impl AVFrame {
    pub fn replace_raw_data(&mut self, data: &[u8]) {
        unsafe {
            self.data[0].copy_from_nonoverlapping(data.as_ptr(), data.len());
        }
    }

    /// Fill the frame with data.
    /// Data format: RGB|RGB|RGB|...
    pub fn fill_data(&mut self, data: &[u8], pix_fmt: AVPixelFormat) {
        let width = if pix_fmt == AVPixelFormat::AvPixFmtGray8 {
            unsafe { *self.inner }.width
        }else {
            unsafe { *self.inner }.width * 3
        };
        let height = unsafe { *self.inner }.height;
        let line_size = self.linesize[0];

        if width % 2 == 0 && height % 2 == 0 {
            unsafe {
                self.data[0].copy_from_nonoverlapping(data.as_ptr(), (width * height) as usize);
            }
            return;
        }

        let ptr = SafeVecPtr(self.data[0]);
        (0..height)
            .into_par_iter()
            .for_each(|y| {
                unsafe {
                    ptr
                        .add((y * line_size) as usize)
                        .copy_from_nonoverlapping(
                            data.as_ptr().add((y * width) as usize),
                            width as usize
                        )
                }
            });
    }
}