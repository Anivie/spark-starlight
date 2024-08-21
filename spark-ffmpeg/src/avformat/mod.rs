use std::collections::HashMap;
use crate::ffi::AVMediaType;

pub mod avformat_context;
pub mod av_stream;
pub mod av_frame_reader;

wrap!(
    AVFormatContext {
        opened: bool,
        scanned_stream: HashMap<AVMediaType, Vec<u32>>,
    } drop avformat_free_context
);