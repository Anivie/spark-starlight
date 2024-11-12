use crate::avcodec::codec::codec_parameter::AVCodecParameters;
use crate::avcodec::{AVCodec, AVCodecContext};
use crate::avframe::AVFrame;
use crate::ffi::{avcodec_alloc_context3, avcodec_open2, avcodec_parameters_to_context, AVDictionary, AVRational, AVStream};
use crate::ffi_enum::AVPixelFormat;
use crate::DeepClone;
use anyhow::{bail, Result};
use std::ptr::{null, null_mut};

impl AVCodecContext {
    pub fn from_stream(codec: &AVCodec, stream: &AVStream, av_dictionary: Option<AVDictionary>) -> Result<Self> {
        let ptr = unsafe {
            avcodec_alloc_context3(codec.inner.cast_const())
        };

        if ptr.is_null() {
            bail!("Failed to allocate codec context.");
        }

        let mut context = AVCodecContext { inner: ptr };
        context.apply_format(stream)?;
        context.open(Some(codec), av_dictionary)?;

        Ok(context)
    }

    pub fn from_frame(codec: &AVCodec, avframe: &AVFrame, av_dictionary: Option<AVDictionary>) -> Result<Self> {
        let ptr = unsafe {
            avcodec_alloc_context3(codec.inner.cast_const())
        };

        if ptr.is_null() {
            bail!("Failed to allocate codec context.");
        }

        let context = {
            let mut context = AVCodecContext { inner: ptr };
            context.width = avframe.width;
            context.height = avframe.height;
            context.pix_fmt = avframe.format as i32;

            context.time_base = AVRational {
                num: 1,
                den: 25,
            };

            context.codec_id = codec.id;
            context.codec = codec.inner;

            context
        };

        context.open(Some(codec), av_dictionary)?;

        Ok(context)
    }

    pub fn copy_into(other: &AVCodecContext) -> Result<Self> {
        let ptr = unsafe {
            avcodec_alloc_context3(null())
        };

        let parameter = AVCodecParameters::from_context(other)?;

        if ptr.is_null() {
            bail!("Failed to allocate codec context.");
        }

        ffmpeg! {
            avcodec_parameters_to_context(ptr, parameter.inner)
        }
        unsafe {
            (*ptr).codec_id = other.codec_id;
            (*ptr).codec = other.codec;
        }
        
        ffmpeg! {
            avcodec_open2(
                ptr,
                (*ptr).codec,
                null_mut()
            )
        }

        Ok(AVCodecContext { inner: ptr })
    }

    pub fn new(size: (i32, i32), format: AVPixelFormat, codec: &AVCodec, av_dictionary: Option<AVDictionary>) -> Result<Self> {
        let ptr = unsafe {
            avcodec_alloc_context3(codec.inner.cast_const())
        };

        if ptr.is_null() {
            bail!("Failed to allocate codec context.");
        }

        let mut context = AVCodecContext { inner: ptr };
        context.width = size.0;
        context.height = size.1;
        context.pix_fmt = format as i32;

        context.time_base = AVRational {
            num: 1,
            den: 25,
        };

        context.codec_id = codec.id;
        context.codec = codec.inner;

        context.open(Some(codec), av_dictionary)?;

        Ok(context)
    }

    fn open(&self, codec: Option<&AVCodec>, av_dictionary: Option<AVDictionary>) -> Result<()> {
        ffmpeg! {
            avcodec_open2(
                self.inner,
                codec.map(|x| x.inner).unwrap_or(self.codec.cast_mut()),
                av_dictionary
                    .map(|mut x| &mut (&mut x as *mut AVDictionary)as *mut *mut AVDictionary)
                    .unwrap_or(null_mut())
            )
        }

        Ok(())
    }
}

impl DeepClone for AVCodecContext {
    fn deep_clone(&self) -> Result<Self>
    where
        Self: Sized
    {
        let new = AVCodecContext::copy_into(self)?;
        Ok(new)
    }
}