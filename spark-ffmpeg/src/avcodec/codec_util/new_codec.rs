use crate::avcodec::{AVCodec, AVCodecContext};
use crate::ffi::AVRational;
use crate::pixformat::AVPixelFormat;
use anyhow::Result;
use rayon::prelude::*;
use std::ops::Deref;

struct SafeVecPtr(*mut u8);
impl Deref for SafeVecPtr {
    type Target = *mut u8;

    fn deref(&self) -> &Self::Target {
        unsafe { &self.0 }
    }
}
unsafe impl Send for SafeVecPtr {}
unsafe impl Sync for SafeVecPtr {}


impl AVCodecContext {
    pub fn new_save(codec: &AVCodec, size: (i32, i32), pixel_format: AVPixelFormat, bitrate: i64) -> Result<Self> {
        AVCodecContext::new_custom(codec, |c| {
            unsafe {
                (*c).bit_rate = bitrate;
                (*c).width = size.0;
                (*c).height = size.1;
                (*c).time_base = AVRational { num: 1, den: 25 };
                (*c).framerate = AVRational { num: 25, den: 1 };
                (*c).gop_size = 10;
                (*c).max_b_frames = 1;
                (*c).pix_fmt = pixel_format as i32;
            }
            c
        })
    }

    /// Fill the data with the frame.
    /// Note: The data should be in the format of RGB|RGB|RGB|RGB|RGB|RGB|RGB|RGB|...
    ///       or in the format of Y|Y|Y|Y|Y|Y|Y|Y|...
    pub fn fill_data(&mut self, data: &[u8]) {
        if self.pix_fmt == AVPixelFormat::AvPixFmtGray8 as i32 {
            self.fill_data_gray(data);
        } else {
            self.fill_data_rgb(data);
        }
    }

    fn fill_data_rgb(&mut self, data: &[u8]) {
        let frame = &self.inner_frame;

        let width = unsafe { *self.inner }.width;
        let height = unsafe { *self.inner }.height;
        let line_size = frame.linesize[0];

        let mut ptr = SafeVecPtr(frame.data[0]);

        data
            .par_iter()
            .enumerate()
            .for_each(|(index, &x)| {
                if index % 3 == 0 {
                    unsafe {
                        *ptr.add(index) = x;
                    }
                }else if index % 3 == 1 {
                    unsafe {
                        *ptr.add(index) = x;
                    }
                }else if index % 3 == 2 {
                    unsafe {
                        *ptr.add(index) = x;
                    }
                }
            });
    }
    fn fill_data_gray(&mut self, data: &[u8]) {
        let frame = &self.inner_frame;

        let width = unsafe { *self.inner }.width;
        let height = unsafe { *self.inner }.height;
        let line_size = frame.linesize[0];

        if width % 2 == 0 && height % 2 == 0 {
            unsafe {
                frame.data[0].copy_from_nonoverlapping(data.as_ptr(), (width * height)as usize);
            }
            return;
        }

        let mut ptr = SafeVecPtr(frame.data[0]);
        (0..height)
            .into_par_iter()
            .for_each(|x| {
                unsafe {
                    ptr
                        .add((x * line_size) as usize)
                        .copy_from_nonoverlapping(
                            data.as_ptr().add((x * width) as usize),
                            width as usize
                        )
                }
            });
    }
}

#[test]
fn tst() {
    let a = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    a.iter().enumerate().for_each(|(i, x)| {
        if i % 3 == 0 {
            print!("r: {}, ", x);
        }
        if i % 3 == 1 {
            print!("g: {}, ", x);
        }
        if i % 3 == 2 {
            print!("b: {}, ", x);
        }
    });
}