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
        &self.0
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

    pub fn fill_data(&mut self, data: &[u8]) {
        let frame = &self.inner_frame;

        let width = if self.pix_fmt == AVPixelFormat::AvPixFmtGray8 as i32 {
            unsafe { *self.inner }.width
        }else {
            unsafe { *self.inner }.width * 3
        };
        let height = unsafe { *self.inner }.height;
        let line_size = frame.linesize[0];

        if width % 2 == 0 && height % 2 == 0 {
            unsafe {
                frame.data[0].copy_from_nonoverlapping(data.as_ptr(), (width * height) as usize);
            }
            return;
        }

        let ptr = SafeVecPtr(frame.data[0]);
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