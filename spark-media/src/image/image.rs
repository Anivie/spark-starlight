use spark_ffmpeg::avcodec::AVCodecContext;
use spark_ffmpeg::avformat::AVFormatContext;
use spark_ffmpeg::avpacket::AVPacket;
use spark_ffmpeg::sws::SwsContext;


#[derive(Debug, Clone, Default)]
pub(super) struct ImageUtil {
    pub(super) sws: Option<SwsContext>,
    pub(super) format: Option<AVFormatContext>,
}

#[derive(Debug, Clone)]
pub struct Image {
    pub(super) packet: Option<AVPacket>,
    pub(super) utils: ImageUtil,

    pub(super) decoder: Option<AVCodecContext>,
    pub(super) encoder: Option<AVCodecContext>,
}