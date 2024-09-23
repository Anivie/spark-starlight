use crate::avcodec::{AVCodec, AVCodecContext, AVCodecContextRaw};
use crate::avframe::AVFrame;
use crate::avpacket::AVPacket;
use crate::ffi::{av_image_get_buffer_size, avcodec_alloc_context3, avcodec_open2, avcodec_parameters_to_context, avcodec_receive_frame, avcodec_receive_packet, avcodec_send_frame, avcodec_send_packet, AVCodecParameters, AVDictionary, AVStream};
use crate::pixformat::AVPixelFormat;
use anyhow::{anyhow, bail, Result};
use std::ptr::null_mut;

impl AVCodecContext {
    pub fn new(codec: &AVCodec, stream: &AVStream, av_dictionary: Option<AVDictionary>) -> Result<Self> {
        let ptr = unsafe {
            avcodec_alloc_context3(codec.inner.cast_const())
        };

        if ptr.is_null() {
            bail!("Failed to allocate codec context.");
        }

        let mut context = AVCodecContext { inner: ptr, inner_frame: AVFrame::new()? };
        context.apply_format(stream)?;
        context.open(codec, av_dictionary)?;

        Ok(context)
    }

    pub(crate) fn new_custom<F>(codec: &AVCodec, mut custom: F) -> Result<Self>
    where
        F: FnMut(*mut AVCodecContextRaw) -> *mut AVCodecContextRaw
    {
        let ptr = unsafe {
            avcodec_alloc_context3(codec.inner.cast_const())
        };
        if ptr.is_null() {
            bail!("Failed to allocate codec context.");
        }

        let ptr = custom(ptr);

        let frame = {
            let mut frame = AVFrame::new()?;
            unsafe {
                frame.set_size((*ptr).width, (*ptr).height);
                frame.format = (*ptr).pix_fmt;
                frame.alloc_image(AVPixelFormat::try_from((*ptr).pix_fmt)?, (*ptr).width, (*ptr).height)?;
            };

            frame
        };

        let context = AVCodecContext { inner: ptr, inner_frame: frame };
        context.open(codec, None)?;

        Ok(context)
    }

    fn open(&self, codec: &AVCodec, av_dictionary: Option<AVDictionary>) -> Result<()> {
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
                format as i32,
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

    fn apply_format(&mut self, frame: &AVStream) -> Result<()> {
        ffmpeg! {
            avcodec_parameters_to_context(self.inner, (*frame).codecpar as *const AVCodecParameters)
        }

        Ok(())
    }

    fn apply_format_with_parameter(&mut self, format_context: &AVCodecParameters) -> Result<()> {
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

    pub fn receive_packet(&self, packet: &mut AVPacket) -> Result<()> {
        ffmpeg! {
            avcodec_receive_packet(self.inner, packet.inner)
        }

        Ok(())
    }

    pub fn receive_frame(&self) -> Result<&AVFrame> {
        ffmpeg! {
            avcodec_receive_frame(self.inner, self.inner_frame.inner)
        }

        Ok(&self.inner_frame)
    }

    pub fn send_frame(&self, frame: Option<&AVFrame>) -> Result<&AVFrame> {
        ffmpeg! {
            avcodec_send_frame(self.inner, frame.map(|x| x.inner).unwrap_or(self.inner_frame.inner))
        }

        Ok(&self.inner_frame)
    }

    pub fn last_frame(&self) -> &AVFrame {
        &self.inner_frame
    }

    pub fn replace_frame(&mut self, frame: AVFrame) {
        self.inner_frame = frame;
    }
}


#[test]
fn test_codec_context() {
    use crate::avformat::avformat_context::OpenFileToAVFormatContext;
    use crate::ffi::AVCodecID_AV_CODEC_ID_PNG;
    use crate::avformat::AVFormatContext;

    let mut format_context = AVFormatContext::open_file("./data/a.png", None).unwrap();
    format_context.video_stream().unwrap().for_each(|(_, x)| {
        let codec = AVCodec::new_decoder(x).unwrap();
        let mut ctx = AVCodecContext::new(&codec, x, None).unwrap();

        let buffer_size = ctx.calculate_buffer_size(AVPixelFormat::AvPixFmtRgb24).unwrap();
        println!("buffer_size: {:?}", buffer_size);
        println!("codec_id: {:?}", ctx.codec_id);
        assert_eq!(ctx.codec_id, AVCodecID_AV_CODEC_ID_PNG);
    });
}