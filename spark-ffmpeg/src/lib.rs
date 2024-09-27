#![allow(dead_code)]

#[macro_use]
mod util;
mod ffi;

pub mod avformat;
pub mod avcodec;
pub mod avframe;
pub mod sws;
pub mod av_mem_alloc;
pub mod avpacket;
pub mod pixel;
pub mod avstream;
pub mod pixformat;

pub trait DeepClone {
    fn deep_clone(&self) -> anyhow::Result<Self>
    where
        Self: Sized;
}