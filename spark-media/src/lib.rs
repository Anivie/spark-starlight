#![cfg_attr(debug_assertions, allow(warnings))]
#![allow(dead_code)]

use hashbrown::HashMap;
use parking_lot::RwLock;
use spark_ffmpeg::avcodec::AVCodec;
use std::sync::LazyLock;

pub mod image;
pub mod filter;

pub use image::image::Image;
pub use spark_ffmpeg::pixel::pixel_formater::RGB;
pub use spark_ffmpeg::DeepClone;

pub use spark_ffmpeg::avfilter_graph::AVFilterGraph;
pub use spark_ffmpeg::ffi_enum::*;

static CODEC: LazyLock<RwLock<HashMap<AVCodecID, AVCodec>>> = LazyLock::new(|| {
    RwLock::new(HashMap::new())
});