use crate::avframe::AVFrame;

mod open_codec;
mod codec_context;

wrap!(
    AVCodecContext {
        inner_frame: AVFrame,
    } drop2 avcodec_free_context,
    AVCodec
);