#![allow(dead_code)]

#[macro_use]
pub mod util;
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
pub mod avfilter_graph;
pub mod avfilter_context;
mod av_rational;

pub trait DeepClone {
    fn deep_clone(&self) -> anyhow::Result<Self>
    where
        Self: Sized;
}

pub trait CloneFrom<T> {
    fn clone_copy_fields(&mut self, other: &T);
}