#![cfg_attr(debug_assertions, allow(warnings))]
#![allow(dead_code)]

use hashbrown::HashMap;
use parking_lot::RwLock;
use spark_ffmpeg::avcodec::AVCodec;
use spark_ffmpeg::avstream::AVCodecID;
use std::sync::LazyLock;

pub mod image;

pub use image::image::Image;
pub use spark_ffmpeg::pixformat::AVPixelFormat;
pub use spark_ffmpeg::pixel::pixel_formater::RGB;

static CODEC: LazyLock<RwLock<HashMap<AVCodecID, AVCodec>>> = LazyLock::new(|| {
    RwLock::new(HashMap::new())
});