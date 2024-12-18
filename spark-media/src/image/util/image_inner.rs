use spark_ffmpeg::avframe::AVFrame;
use spark_ffmpeg::avpacket::AVPacket;

#[derive(Debug, Default)]
pub(crate) struct ImageInner {
    pub(crate) packet: Option<AVPacket>,
    pub(crate) frame: AVFrame,
}
