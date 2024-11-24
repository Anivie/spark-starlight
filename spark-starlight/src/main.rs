#![feature(let_chains)]
#![cfg_attr(debug_assertions, allow(warnings))]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Result;
use spark_inference::engine::inference_engine::{ExecutionProvider, OnnxSession};
use spark_inference::inference::yolo::inference_yolo_seg::YoloSegmentInference;
use spark_inference::inference::sam::former_state::InferenceType;
use spark_inference::inference::sam::image_inference::{SAM2ImageInferenceSession, SamImageInference};
use spark_inference::inference::sam::video_inference::{SAM2VideoInferenceSession, SamVideoInference};
use spark_inference::inference::yolo::inference_yolo_detect::{YoloDetectInference, YoloDetectSession};
use spark_inference::utils::graph::Point;
use spark_inference::utils::masks::ApplyMask;
use spark_media::{Image, RGB};
use spark_media::filter::filter::AVFilter;

fn main_yolo_seg() -> Result<()> {
    let path = "./data/image/a.png";
    let image = Image::open_file(path)?;

    let yolo = OnnxSession::new("./data/model/yolo_sam.onnx", ExecutionProvider::CUDA)?;

    let results = yolo.inference_yolo(image, 0.25, 0.45)?;
    println!("{:?}", results.len());

    let mut image = Image::open_file(path)?;
    let filter = AVFilter::builder(image.pixel_format()?, image.get_size())?
        .add_context("scale", "640:640:force_original_aspect_ratio=decrease")?
        .add_context("pad", "640:640:(ow-iw)/2:(oh-ih)/2:#727272")?
        .add_context("format", "rgb24")?
        .build()?;
    image.apply_filter(&filter)?;

    for result in results {
        image.layering_mask(&result.mask, RGB(20, 20, 0))?;
    }

    image.save_with_format("./data/out/y_out.png")?;

    Ok(())
}

fn main_sam() -> Result<()> {
    let sam2 = SAM2VideoInferenceSession::new("./data/model")?;
    // let sam2 = SAM2VideoInferenceSession::new("/home/git/SAM2Export-origin/checkpoints/tiny")?;

    let path = "./data/image/bird1.png";
    let mut image = Image::open_file(path)?;

    let result = sam2.inference_sam(
        InferenceType::First(vec![Point { x: 895., y: 554. }, Point { x: 677., y: 648. }]),
        &mut image
    )?;

    let mut image = Image::open_file(path)?;
    let filter = AVFilter::builder(image.pixel_format()?, image.get_size())?
        .add_context("format", "rgb24")?
        .build()?;

    image.apply_filter(&filter)?;
    image.layering_mask(&result.mask, RGB(0, 50, 0))?;
    image.save_with_format("./data/out/a_out.png")?;
    //**********************************************************************************
    let path = "./data/image/bird2.png";
    let mut image = Image::open_file(path)?;

    let result = sam2.inference_sam(
        InferenceType::WithState(result.state),
        &mut image
    )?;

    let mut image = Image::open_file(path)?;
    let filter = AVFilter::builder(image.pixel_format()?, image.get_size())?
        .add_context("format", "rgb24")?
        .build()?;

    image.apply_filter(&filter)?;
    image.layering_mask(&result.mask, RGB(50, 0, 0))?;
    image.save_with_format("./data/out/b_out.png")?;
    //**********************************************************************************
    let path = "./data/image/bird3.png";
    let mut image = Image::open_file(path)?;

    let result = sam2.inference_sam(
        InferenceType::WithState(result.state),
        &mut image
    )?;

    let mut image = Image::open_file(path)?;
    let filter = AVFilter::builder(image.pixel_format()?, image.get_size())?
        .add_context("format", "rgb24")?
        .build()?;

    image.apply_filter(&filter)?;
    image.layering_mask(&result.mask, RGB(0, 0, 50))?;
    image.save_with_format("./data/out/c_out.png")?;

    Ok(())
}

fn main_sam_image() -> Result<()> {
    let sam2 = SAM2ImageInferenceSession::new("/home/git/SAM2Export-origin/checkpoints/tiny")?;

    let path = "./data/image/bird1.png";
    let image = Image::open_file(path)?;

    let result = sam2.inference_sam(vec![Point { x: 92., y: 983. }], image)?;

    let mut image = Image::open_file(path)?;
    let filter = AVFilter::builder(image.pixel_format()?, image.get_size())?
        .add_context("format", "rgb24")?
        .build()?;

    image.apply_filter(&filter)?;
    image.layering_mask(&result, RGB(200, 0, 0))?;
    image.save_with_format("./data/out/a_out.png")?;

    Ok(())
}

fn main() -> Result<()> {
    let yolo = YoloDetectSession::new("./data/model")?;
    let sam2 = SAM2ImageInferenceSession::new("/home/git/SAM2Export-origin/checkpoints/tiny")?;

    let path = "./data/image/rt.jpeg";
    let image = Image::open_file(path)?;

    let results = yolo.inference_yolo(image, 0.75)?;
    println!("results: {:?}", results);

    let image = Image::open_file(path)?;
    let result = sam2.inference_sam(
        results.iter().map(|x| Point{ x: x.x, y: x.y }).collect(),
        image
    )?;

    let mut image = Image::open_file(path)?;
    let filter = AVFilter::builder(image.pixel_format()?, image.get_size())?
        .add_context("format", "rgb24")?
        .build()?;

    image.apply_filter(&filter)?;
    image.layering_mask(&result, RGB(200, 0, 0))?;
    image.save_with_format("./data/out/a_out.png")?;


    Ok(())
}