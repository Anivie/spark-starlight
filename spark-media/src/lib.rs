#![feature(core_intrinsics)]

use hashbrown::HashMap;
use parking_lot::RwLock;
use spark_ffmpeg::avcodec::{AVCodec, AVCodecContext};
use spark_ffmpeg::avformat::AVFormatContext;
use spark_ffmpeg::avstream::AVCodecID;
use std::sync::LazyLock;

pub struct Image {
    format: Option<AVFormatContext>,
    codec: AVCodecContext,
}

static CODEC: LazyLock<RwLock<HashMap<AVCodecID, AVCodec>>> = LazyLock::new(|| {
    RwLock::new(HashMap::new())
});


pub mod decoder;
pub mod image_util;
pub mod encoder;