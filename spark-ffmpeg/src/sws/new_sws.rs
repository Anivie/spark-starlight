use crate::avcodec::AVCodecContext;
use crate::avframe::AVFrame;
use crate::ffi::{sws_getContext, sws_scale, SWS_BILINEAR};
use crate::pixformat::AVPixelFormat;
use crate::sws::SwsContext;
use anyhow::{bail, Result};
use std::ptr::null_mut;

impl SwsContext {
    pub fn from_format_context(codec_context: &AVCodecContext, dst_format: Option<AVPixelFormat>, dst_size: Option<(i32, i32)>, flags: Option<u32>) -> Result<Self> {
        let sws = unsafe {
            sws_getContext(
                codec_context.width,
                codec_context.height,
                codec_context.pix_fmt,
                dst_size.map(|(x, _)| x).unwrap_or(codec_context.width),
                dst_size.map(|(_, y)| y).unwrap_or(codec_context.height),
                dst_format.map(|x| x as i32).unwrap_or(codec_context.pix_fmt),
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
        src: &AVFrame,
        dst: &AVFrame
    ) -> Result<i32> {
        let slice_height = unsafe {
            sws_scale(
                self.inner,
                src.data.as_ptr() as *const *const u8,
                src.linesize.as_ptr(),
                0,
                src.height,
                dst.data.as_ptr(),
                dst.linesize.as_ptr(),
            )
        };

        Ok(slice_height)
    }
}

#[test]
fn test_sws_context() {
    use crate::avcodec::{AVCodec, AVCodecContext};
    use crate::avformat::AVFormatContext;
    use crate::avformat::avformat_context::OpenFileToAVFormatContext;

    let mut av_format_context = AVFormatContext::open_file("./data/a.png", None).unwrap();
    av_format_context.video_stream().unwrap().for_each(|(_, x)| {
        let codec = AVCodec::new_decoder(x).unwrap();
        let av_codec_context = AVCodecContext::from_stream(&codec, x, None).unwrap();
        let _ = SwsContext::from_format_context(&av_codec_context, None, None, None).unwrap();
    });
}