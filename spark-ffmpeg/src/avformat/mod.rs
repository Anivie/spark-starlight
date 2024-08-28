use std::collections::HashMap;

pub mod avformat_context;
pub mod av_stream;
pub mod av_frame_reader;

#[repr(i32)]
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum AVMediaType {
    UNKNOWN = -1,
    VIDEO = 0,
    AUDIO = 1,
    DATA = 2,
    SUBTITLE = 3,
    ATTACHMENT = 4,
    NB = 5,
}

wrap!(
    AVFormatContext {
        opened: bool,
        scanned_stream: HashMap<AVMediaType, Vec<u32>>,
    } drop avformat_free_context
);