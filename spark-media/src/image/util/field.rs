use spark_ffmpeg::avframe::AVFrame;
use crate::Image;

impl Image {
    pub(crate) fn frame(&self) -> &AVFrame {
        self.codec.last_frame()
    }
}