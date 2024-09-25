use crate::avformat::{AVFormatContext, AVFormatContextRaw};
use crate::avpacket::AVPacket;
use crate::ffi::{av_read_frame, avformat_alloc_context, avformat_open_input, AVDictionary, AVInputFormat};
use anyhow::Result;
use std::ffi::CString;
use std::ptr::{null, null_mut};
use num_enum::TryFromPrimitiveError;
use crate::pixformat::AVPixelFormat;

pub trait OpenFileToAVFormatContext {
    fn open_file(path: impl Into<String>, format: Option<&AVInputFormat>) -> Result<Self>
    where
        Self: Sized;
    fn open_file_arg<'a>(path: &str, format: Option<&AVInputFormat>, dictionary: &'a mut AVDictionary) -> Result<(Self, &'a AVDictionary)>
    where
        Self: Sized;
}

impl OpenFileToAVFormatContext for AVFormatContext {
    fn open_file(path: impl Into<String>, format: Option<&AVInputFormat>) -> Result<Self> {
        let mut av_format_context = unsafe {
            avformat_alloc_context()
        };
        let path = CString::new(path.into().as_str())?;

        ffmpeg! {
            avformat_open_input(
                &mut av_format_context as *mut *mut AVFormatContextRaw,
                path.as_ptr(),
                format
                    .map(|x| x as *const AVInputFormat)
                    .unwrap_or_else(|| null::<AVInputFormat>()),
                null_mut::<*mut AVDictionary>()
            ) or "Failed to open file"
        }

        Ok(AVFormatContext{
            inner: av_format_context,
            opened: false,
            scanned_stream: Default::default(),
        })
    }

    fn open_file_arg<'a>(path: &str, format: Option<&AVInputFormat>, dictionary: &'a mut AVDictionary) -> Result<(Self, &'a AVDictionary)>
    where
        Self: Sized
    {
        let mut av_format_context = unsafe {
            avformat_alloc_context()
        };
        let path = CString::new(path)?;

        ffmpeg! {
            avformat_open_input(
                &mut av_format_context as *mut *mut AVFormatContextRaw,
                path.as_ptr(),
                format.map(|x| x as *const AVInputFormat).unwrap_or_else(|| null::<AVInputFormat>()),
                &mut (dictionary as *mut AVDictionary)as *mut *mut AVDictionary
            ) or "Failed to open file"
        }

        Ok((AVFormatContext{
            inner: av_format_context,
            opened: false,
            scanned_stream: Default::default(),
        }, dictionary))
    }
}

impl AVFormatContext {
    pub fn read_frame(&self, packet: &mut AVPacket) -> Result<()> {
        native!(
            av_read_frame(self.inner, packet.inner) or "Failed to read frame"
        );

        Ok(())
    }

    pub fn pixel_format(&self, stream_index: usize) -> Result<AVPixelFormat> {
        let format = unsafe {
            (*(**(*self.inner).streams.add(stream_index)).codecpar).format
        };

        Ok(AVPixelFormat::try_from(format)?)
    }
}

#[test]
fn test_video_stream() {
    use crate::ffi::AVMediaType_AVMEDIA_TYPE_VIDEO;
    let mut a = AVFormatContext::open_file("./data/a.png", None).unwrap();
    let stream = a.video_stream().unwrap();
    stream.for_each(|(_, x)| {
        unsafe { println!("{:?}", (*x.codecpar).codec_type); }
        unsafe { assert_eq!((*x.codecpar).codec_type, AVMediaType_AVMEDIA_TYPE_VIDEO) }
    });
}

#[test]
fn test_format() {
    use crate::avformat::AVMediaType;
    let mut a = AVFormatContext::open_file("./data/a.png", None).unwrap();
    a.find_stream(AVMediaType::VIDEO).and_then(|x| {
        println!("{:?}", x);
        Ok(())
    }).unwrap();
    println!("a: {}", a.nb_streams);
}