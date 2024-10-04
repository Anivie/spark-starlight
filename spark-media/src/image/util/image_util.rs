use spark_ffmpeg::avformat::AVFormatContext;
use spark_ffmpeg::sws::SwsContext;

#[derive(Debug, Default)]
pub(crate) struct ImageUtil {
    pub(crate) sws: Option<SwsContext>,
    pub(crate) format: Option<AVFormatContext>,
}