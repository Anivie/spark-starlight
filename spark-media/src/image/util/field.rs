use std::mem::ManuallyDrop;
use spark_ffmpeg::avcodec::AVCodecContext;
use spark_ffmpeg::avframe::AVFrame;
use spark_ffmpeg::avstream::AVCodecID;
use spark_ffmpeg::pixformat::AVPixelFormat;
use crate::Image;

impl Image {
    pub(crate) fn available_codec(&self) -> &AVCodecContext {
        self.encoder.as_ref()
            .unwrap_or_else(|| self.decoder.as_ref().expect("At least one codec must be available"))
    }

    pub fn frame(&self) -> &AVFrame {
        self.available_codec().last_frame()
    }

    pub fn pixel_format(&self) -> AVPixelFormat {
        self.available_codec().pixel_format()
    }

    pub fn codec_id(&self) -> AVCodecID {
        self.available_codec().id()
    }

    pub fn raw_data(&self) -> ManuallyDrop<Vec<u8>> {
        self.available_codec().last_frame().get_raw_data(0)
    }
}