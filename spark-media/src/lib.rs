use hashbrown::HashMap;
use parking_lot::RwLock;
use spark_ffmpeg::avcodec::{AVCodec, AVCodecContext};
use spark_ffmpeg::avformat::AVFormatContext;
use spark_ffmpeg::avstream::AVCodecID;
use std::sync::LazyLock;

pub mod decoder;
pub mod image_util;
pub mod encoder;

pub use spark_ffmpeg::pixformat::AVPixelFormat;
use spark_ffmpeg::sws::SwsContext;

pub struct Image {
    sws: Option<SwsContext>,
    format: Option<AVFormatContext>,
    codec: AVCodecContext,
}

static CODEC: LazyLock<RwLock<HashMap<AVCodecID, AVCodec>>> = LazyLock::new(|| {
    RwLock::new(HashMap::new())
});