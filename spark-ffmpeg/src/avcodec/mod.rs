use spark_proc_macro::wrap_ffmpeg;

pub mod codec;
mod codec_util;

wrap_ffmpeg!(AVCodec);

wrap_ffmpeg!(
    AVCodecContext drop+ [avcodec_free_context]
);

unsafe impl Send for AVCodec {}
unsafe impl Sync for AVCodec {}
