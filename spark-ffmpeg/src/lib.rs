#![allow(warnings)]

use std::fs::File;
use std::io::Write;
use std::mem::ManuallyDrop;
use crate::ffi::{av_image_alloc, av_malloc};
use image::{ColorType, DynamicImage, GenericImage, GenericImageView, Pixel, Rgba};
use image::imageops::FilterType;
use ndarray::Array4;
use crate::avframe::AVFrame;
use crate::pixel::frames::AVFrameCollector;
use crate::sws::SwsContext;

#[macro_use]
mod util;
mod ffi;


pub mod avformat;
pub mod avcodec;
pub mod avframe;
pub mod sws;
pub mod av_mem_alloc;
pub mod avpacket;
pub mod pixel;

pub fn get_pixels() -> anyhow::Result<Vec<f32>> {
    use crate::av_mem_alloc::AVMemorySegment;
    use crate::avcodec::{AVCodec, AVCodecContext};
    use crate::avformat::avformat_context::OpenFileToAVFormatContext;
    use crate::avformat::AVFormatContext;
    use crate::ffi::{ AVPixelFormat_AV_PIX_FMT_RGB24, AVMediaType_AVMEDIA_TYPE_VIDEO };

    // let mut av_format_context = AVFormatContext::open_file("/mnt/c/Users/anivi/OneDrive/Videos/Desktop/r.mp4", None)?;
    let mut av_format_context = AVFormatContext::open_file("/home/spark-starlight/data/image/c.jpg", None)?;
    let stream = av_format_context.video_stream()?;
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
            av_codec_context
        })
        .collect::<Vec<_>>();
    let av_codec_context = codecs.remove(0);

    let mut vec = av_format_context
        .frames(AVMediaType_AVMEDIA_TYPE_VIDEO)
        ?
        .map(|mut packet| {
            av_codec_context.send_packet(&mut packet).unwrap();
            av_codec_context.receive_frame().unwrap()
        })
        .collect::<Vec<_>>();

    let sws = SwsContext::from_format_context(&av_codec_context, Some(AVPixelFormat_AV_PIX_FMT_RGB24), Some((640, 640)), None)?;
    let scaled_frame = {
        let mut scaled_frame = AVFrame::new()?;
        scaled_frame.width = 640;
        scaled_frame.height = 640;
        scaled_frame.alloc_image(AVPixelFormat_AV_PIX_FMT_RGB24, 640, 640)?;
        scaled_frame
    };
    sws.scale_image(&vec[0], &scaled_frame)?;

    let data = unsafe {
        let size = (scaled_frame.height * scaled_frame.linesize[0]) as usize;
        ManuallyDrop::new(Vec::from_raw_parts(scaled_frame.data[0], size, size))
    };
    let mut r: Vec<f32> = data.iter()
        .enumerate()
        .filter(|(index, _)| *index % 3 == 0)
        .map(|(_, x)| *x as f32 / 255.)
        .collect();
    let g: Vec<f32> = data.iter()
        .enumerate()
        .filter(|(index, _)| (*index + 2) % 3 == 0)
        .map(|(_, x)| *x as f32 / 255.)
        .collect();
    let b: Vec<f32> = data.iter()
        .enumerate()
        .filter(|(index, _)| (*index + 1) % 3 == 0)
        .map(|(_, x)| *x as f32 / 255.)
        .collect();
    r.extend_from_slice(&g);
    r.extend_from_slice(&b);

    Ok(r)
}

#[test]
fn test_array_filter() {
    //                 R  G  B  R  G  B  R  G  B
    let data = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    data.iter()
        .enumerate()
        .filter(|(index, _)| {
            (index + 2) % 3 == 0
        })
        .map(|(index, x)| *x as f32)
        .for_each(|x| {
            println!("{}", x);
        });
}