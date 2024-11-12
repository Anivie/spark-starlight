use crate::image::util::image_inner::ImageInner;
use crate::image::util::image_util::ImageUtil;
use spark_ffmpeg::avcodec::AVCodecContext;
use spark_ffmpeg::DeepClone;

#[derive(Debug)]
pub struct Image {
    pub(super) inner: ImageInner,
    pub(super) utils: ImageUtil,

    pub(super) decoder: Option<AVCodecContext>,
    pub(super) encoder: Option<AVCodecContext>,
}

impl Clone for Image {
    fn clone(&self) -> Self {
        Self {
            utils: Default::default(),
            inner: ImageInner {
                packet: None,
                frame: self.inner.frame.deep_clone().unwrap(),
            },
            decoder: None,
            encoder: None,
        }
    }
}