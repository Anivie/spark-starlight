#![feature(let_chains)]
#![cfg_attr(debug_assertions, allow(warnings))]

use hashbrown::HashMap;
use parking_lot::RwLock;
use spark_ffmpeg::avcodec::AVCodec;
use std::sync::LazyLock;

pub mod filter;
pub mod image;

pub use image::image::Image;
pub use spark_ffmpeg::pixel::pixel_formater::RGB;
pub use spark_ffmpeg::DeepClone;

pub use spark_ffmpeg::avfilter_graph::AVFilterGraph;
pub use spark_ffmpeg::ffi_enum::*;

static CODEC: LazyLock<RwLock<HashMap<AVCodecID, AVCodec>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));
