use crate::image::util::inner_lock::InnerLock;
use spark_ffmpeg::avcodec::AVCodecContext;
use spark_ffmpeg::avformat::AVFormatContext;
use spark_ffmpeg::avframe::AVFrame;
use spark_ffmpeg::avpacket::AVPacket;
use spark_ffmpeg::sws::SwsContext;

#[derive(Debug, Clone, Default)]
pub(super) struct ImageUtil {
    pub(super) sws: Option<SwsContext>,
    pub(super) format: Option<AVFormatContext>,
}

#[derive(Debug, Clone, Default)]
pub(super) struct ImageInner {
    pub(super) packet: Option<InnerLock<AVPacket>>,
    pub(super) frame: Option<InnerLock<AVFrame>>,
}

#[derive(Debug, Clone)]
pub struct Image {
    pub(super) inner: ImageInner,
    pub(super) utils: ImageUtil,

    pub(super) decoder: Option<AVCodecContext>,
    pub(super) encoder: Option<AVCodecContext>,
}

impl ImageInner {
    pub(crate) fn replace_packet(&mut self, packet: AVPacket) {
        self.packet.replace(InnerLock::new(packet));
    }

    pub(crate) fn replace_frame(&mut self, frame: AVFrame) {
        self.frame.replace(InnerLock::new(frame));
    }
}