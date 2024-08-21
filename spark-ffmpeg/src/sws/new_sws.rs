use crate::avcodec::{AVCodec, AVCodecContext};
use crate::ffi::{sws_getContext, sws_scale, AVPixelFormat, SWS_BILINEAR};
use crate::sws::SwsContext;
use anyhow::{bail, Result};
use std::ptr::null_mut;
use crate::avformat::avformat_context::OpenFileToAVFormatContext;
use crate::avformat::AVFormatContext;
use crate::avframe::AVFrame;

impl SwsContext {
    pub fn from_format_context(codec_context: &AVCodecContext, dst_format: Option<AVPixelFormat>, flags: Option<u32>) -> Result<Self> {
        let sws = unsafe {
            sws_getContext(
                codec_context.width,
                codec_context.height,
                codec_context.pix_fmt,
                codec_context.width,
                codec_context.height,
                dst_format.unwrap_or(codec_context.pix_fmt),
                flags.map(|x| x as i32).unwrap_or(SWS_BILINEAR as i32),
                null_mut(),
                null_mut(),
                null_mut(),
            )
        };

        if sws.is_null() {
            bail!("Failed to create SwsContext");
        }

        Ok(SwsContext { inner: sws })
    }

    pub fn scale_image(
        &self,
        avcodec_context: &AVCodecContext,
        src: &AVFrame,
        dst: &AVFrame
    ) -> Result<i32> {
        let slice_height = unsafe {
            sws_scale(
                self.inner,
                src.data.as_ptr() as *const *const u8,
                src.linesize.as_ptr(),
                0,
                avcodec_context.height,
                dst.data.as_ptr(),
                dst.linesize.as_ptr(),
            )
        };

        Ok(slice_height)
    }
}

#[test]
fn test_sws_context() {
    let mut av_format_context = AVFormatContext::open_file("./data/a.png", None).unwrap();
    av_format_context.video_stream().unwrap().for_each(|(_, x)| {
        let codec = unsafe {
            AVCodec::open_codec((*x.codecpar).codec_id).unwrap()
        };
        let av_codec_context = unsafe {
            let mut av_codec_context = AVCodecContext::new(Some(&codec), None).unwrap();
            av_codec_context.apply_format(&*x.codecpar).unwrap();
            av_codec_context
        };
        let _ = SwsContext::from_format_context(&av_codec_context, None, None).unwrap();
    });
}