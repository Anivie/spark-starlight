pub mod codec;
mod codec_util;

wrap!(
    AVCodecContext drop2 avcodec_free_context,
    AVCodec
);

unsafe impl Send for AVCodec {}
unsafe impl Sync for AVCodec {}