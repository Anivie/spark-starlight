use crate::avframe::AVFrame;

pub mod codec_alloc;
mod codec_util;

wrap!(
    AVCodecContext {
        inner_frame: AVFrame,
    } drop2 avcodec_free_context,
    AVCodec
);

unsafe impl Send for AVCodec {}
unsafe impl Sync for AVCodec {}