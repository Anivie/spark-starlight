#![allow(dead_code)]

#[macro_use]
pub mod util;
mod ffi;

pub mod av_mem_alloc;
mod av_rational;
pub mod avcodec;
pub mod avfilter_context;
pub mod avfilter_graph;
pub mod avformat;
pub mod avframe;
pub mod avpacket;
pub mod avstream;
pub mod ffi_enum;
pub mod pixel;
pub mod sws;

pub trait DeepClone {
    fn deep_clone(&self) -> anyhow::Result<Self>
    where
        Self: Sized;
}

pub trait CloneFrom<T> {
    fn clone_fields_from(&mut self, other: &T);
}

pub fn disable_ffmpeg_logging() {
    unsafe {
        ffi::av_log_set_level(ffi::AV_LOG_QUIET);
    }
}
