use crate::avcodec::{AVCodec, AVCodecContext, AVCodecRaw};
use crate::ffi::{av_image_get_buffer_size, avcodec_alloc_context3, avcodec_open2, avcodec_parameters_to_context, avcodec_receive_frame, avcodec_send_packet, AVCodecID_AV_CODEC_ID_PNG, AVCodecParameters, AVDictionary, AVPixelFormat, AVPixelFormat_AV_PIX_FMT_RGB24};
use anyhow::{anyhow, Result};
use std::ptr::{null, null_mut};
use crate::avformat::AVFormatContext;
use crate::avframe::AVFrame;
use crate::avpacket::AVPacket;

impl AVCodecContext {
    pub fn new(codec: Option<&AVCodec>, av_dictionary: Option<AVDictionary>) -> Result<Self> {
        let codec_ptr = codec.map(|c| c.inner.cast_const()).unwrap_or(null::<AVCodecRaw>());

        let ptr = unsafe {
            avcodec_alloc_context3(codec_ptr)
        };

        Ok(AVCodecContext { inner: ptr, inner_frame: AVFrame::new()? })
    }

    pub fn open(&self, codec: &AVCodec, av_dictionary: Option<AVDictionary>) -> Result<()> {
        native! {
            avcodec_open2(
                self.inner,
                codec.inner.cast_const(),
                av_dictionary
                    .map(|mut x| &mut (&mut x as *mut AVDictionary)as *mut *mut AVDictionary)
                    .unwrap_or(null_mut())
            )
        };

        Ok(())
    }

    pub fn calculate_buffer_size(&self, format: AVPixelFormat) -> Result<i32> {
        let size = unsafe {
            av_image_get_buffer_size(
                format,
                self.width,
                self.height,
                1,
            )
        };

        if size > 0 {
            Ok(size)
        } else {
            Err(anyhow!("Failed to get buffer size with code {}.", size))
        }
    }

    pub fn apply_format(&mut self, format_context: &AVCodecParameters) -> Result<()> {
        ffmpeg! {
            avcodec_parameters_to_context(self.inner, format_context as *const AVCodecParameters)
        }

        Ok(())
    }

    pub fn send_packet(&self, packet: &mut AVPacket) -> Result<()> {
        ffmpeg! {
            avcodec_send_packet(self.inner, packet.inner)
        }

        Ok(())
    }

    pub fn receive_frame(&self) -> Result<&AVFrame> {
        ffmpeg! {
            avcodec_receive_frame(self.inner, self.inner_frame.inner)
        }

        Ok(&self.inner_frame)
    }
}


#[test]
fn test_codec_context() {
    use crate::avformat::avformat_context::OpenFileToAVFormatContext;

    let mut format_context = AVFormatContext::open_file("./data/a.png", None).unwrap();
    format_context.video_stream().unwrap().for_each(|(_, x)| {
        let codec = AVCodec::open_codec(AVCodecID_AV_CODEC_ID_PNG).unwrap();
        let mut ctx = AVCodecContext::new(Some(&codec), None).unwrap();

        unsafe {
            ctx.apply_format(&*x.codecpar).unwrap();
        }

        let buffer_size = ctx.calculate_buffer_size(AVPixelFormat_AV_PIX_FMT_RGB24).unwrap();
        println!("buffer_size: {:?}", buffer_size);
        println!("codec_id: {:?}", ctx.codec_id);
        assert_eq!(ctx.codec_id, AVCodecID_AV_CODEC_ID_PNG);
    });
}