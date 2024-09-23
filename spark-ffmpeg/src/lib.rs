#![allow(dead_code)]

use crate::avformat::AVMediaType;
use crate::avframe::AVFrame;
use crate::pixformat::AVPixelFormat;
use crate::sws::SwsContext;
use anyhow::Result;
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
pub mod pixel;
pub mod avstream;
pub mod pixformat;

pub fn get_pixels() -> Result<Vec<f32>> {
    use crate::avcodec::{AVCodec, AVCodecContext};
    use crate::avformat::avformat_context::OpenFileToAVFormatContext;
    use crate::avformat::AVFormatContext;
    // let mut av_format_context = AVFormatContext::open_file("/mnt/c/Users/anivi/OneDrive/Videos/Desktop/r.mp4", None)?;
    let mut av_format_context = AVFormatContext::open_file("/home/spark-starlight/data/image/b.png", None)?;
    let stream = av_format_context.video_stream()?;
    let mut codecs = stream
        .map(|(_, av_stream)| {
            let codec = AVCodec::new_decoder(av_stream).unwrap();
            let av_codec_context = AVCodecContext::new(&codec, av_stream, None).unwrap();

            av_codec_context
        })
        .collect::<Vec<_>>();
    let av_codec_context = codecs.remove(0);

    let vec = av_format_context
        .frames(AVMediaType::VIDEO)?
        .map(|mut packet| {
            av_codec_context.send_packet(&mut packet).unwrap();
            av_codec_context.receive_frame().unwrap()
        })
        .collect::<Vec<_>>();

    let sws = SwsContext::from_format_context(&av_codec_context, Some(AVPixelFormat::AvPixFmtRgb24), Some((640, 640)), None)?;
    let scaled_frame = {
        let mut scaled_frame = AVFrame::new()?;
        scaled_frame.width = 640;
        scaled_frame.height = 640;
        scaled_frame.alloc_image(AVPixelFormat::AvPixFmtRgb24, 640, 640)?;
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