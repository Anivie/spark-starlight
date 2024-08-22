#![allow(dead_code)]

use std::mem::ManuallyDrop;

#[macro_use]
mod util;
mod ffi;


pub mod avformat;
pub mod avcodec;
pub mod avframe;
pub mod sws;
pub mod av_mem_alloc;
pub mod avpacket;

fn get_pixels() -> Vec<ManuallyDrop<Vec<u8>>> {
    use crate::av_mem_alloc::AVMemorySegment;
    use crate::avcodec::{AVCodec, AVCodecContext};
    use crate::avformat::avformat_context::OpenFileToAVFormatContext;
    use crate::avformat::AVFormatContext;
    use crate::sws::SwsContext;
    use crate::avframe::AVFrame;
    use crate::ffi::AVPixelFormat_AV_PIX_FMT_RGB24;
    use ffi::AVMediaType_AVMEDIA_TYPE_VIDEO;
    use std::mem::forget;

    // let mut av_format_context = AVFormatContext::open_file("/mnt/c/Users/anivi/OneDrive/Videos/Desktop/r.mp4", None).unwrap();
    let mut av_format_context = AVFormatContext::open_file(".././data/image/a.png", None).unwrap();
    let stream = av_format_context.video_stream().unwrap();
    let mut codecs = stream
        .map(|(stream_index, av_stream)| {
            let codec = unsafe {
                AVCodec::open_codec((*av_stream.codecpar).codec_id).unwrap()
            };

            let av_codec_context = unsafe {
                let mut av_codec_context = AVCodecContext::new(Some(&codec), None).unwrap();
                av_codec_context.apply_format(&*av_stream.codecpar).unwrap();
                av_codec_context.open(&codec, None).unwrap();
                av_codec_context
            };

            let segment = {
                let size = av_codec_context.calculate_buffer_size(AVPixelFormat_AV_PIX_FMT_RGB24).unwrap();
                let segment = AVMemorySegment::new(size as usize).unwrap();
                segment
            };
            (codec, av_codec_context, segment, stream_index)
        })
        .collect::<Vec<_>>();
    let (codec, av_codec_context, segment, stream_index) = codecs.remove(0);
    println!("segment: {:?}", segment.size);

    let rgb_frame = {
        let mut rgb_frame = AVFrame::new().unwrap();
        rgb_frame.fill_arrays(
            segment.inner.cast::<u8>(),
            AVPixelFormat_AV_PIX_FMT_RGB24,
            av_codec_context.width,
            av_codec_context.height,
        ).unwrap();
        rgb_frame
    };
    forget(segment);

    let sws = SwsContext::from_format_context(&av_codec_context, Some(AVPixelFormat_AV_PIX_FMT_RGB24), None).unwrap();
    let mut vec = av_format_context
        .frames(AVMediaType_AVMEDIA_TYPE_VIDEO)
        .unwrap()
        .map(|mut packet| {
            av_codec_context.send_packet(&mut packet).unwrap();
            let frame = av_codec_context.receive_frame().unwrap();

            sws.scale_image(&av_codec_context, frame, &rgb_frame).unwrap();

            let base_addr = rgb_frame.data.get(0).unwrap();
            let line_size = *rgb_frame.linesize.get(0).unwrap();
            let size = (av_codec_context.width * line_size)as usize;
            println!("size: {}", size);
            let vec = unsafe {
                ManuallyDrop::new(Vec::<u8>::from_raw_parts(*base_addr, size, size))
            };

            vec
        })
        .collect::<Vec<_>>();

    vec
}

#[test]
fn test_get_pixels() {
    let mut vec = get_pixels();
    println!("len: {:?}", vec.len());
    println!("{}", vec[0][0]);
    println!("found: {} packet", vec.len());
    let i = vec.len();
    vec[i - 1][0] = 0;
    println!("set: {}", vec[i - 1][0])
}