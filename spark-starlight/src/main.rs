#![feature(let_chains)]
#![cfg_attr(debug_assertions, allow(warnings))]

use anyhow::Result;
use ndarray::{array, concatenate, s, Axis};
use spark_inference::engine::inference_engine::{ExecutionProvider, OnnxSession};
use spark_inference::inference::inference_yolo::YoloModelInference;
use spark_inference::inference::sam_former_state::InferenceType;
use spark_inference::inference::sam_image_inference::{SAM2ImageInferenceSession, SamImageInference};
use spark_inference::inference::sam_video_inference::{SAM2VideoInferenceSession, SamVideoInference};
use spark_inference::utils::graph::Point;
use spark_inference::utils::masks::ApplyMask;
use spark_media::{Image, RGB};
use spark_media::filter::filter::AVFilter;

fn main_yolo() -> Result<()> {
    let path = "./data/image/a.png";
    let image = Image::open_file(path)?;

    let yolo = OnnxSession::new("./data/model/best2.onnx", ExecutionProvider::CUDA)?;

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

fn maind() -> Result<()> {
    let arr1 = array![
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.]
    ];
    let arr2 = array![
            [10., 11., 12.],
            [13., 14., 15.],
            [16., 17., 18.]
    ];

    let arr3 = concatenate![Axis(0), arr1, arr2];
    println!("{:?}", arr3);
    println!("{:?}", arr3.shape()[0]);
    println!("{:?}", arr3.shape()[1]);

    if arr3.shape()[0] >= 6 {
        let arr4 = arr3.slice(s![1.., ..]).to_owned();
        println!("{:?}", arr4);
    }

    Ok(())
}
fn main() -> Result<()> {
    let sam2 = SAM2VideoInferenceSession::new("./data/model")?;

    let path = "./data/image/brid1.png";
    let mut image = Image::open_file(path)?;

    let result = sam2.inference_sam(
        InferenceType::First(vec![Point { x: 96, y: 984 }]),
        &mut image
    )?;

    let mut image = Image::open_file(path)?;
    let filter = AVFilter::builder(image.pixel_format()?, image.get_size())?
        .add_context("format", "rgb24")?
        .build()?;

    image.apply_filter(&filter)?;
    image.layering_mask(&result.mask, RGB(200, 0, 0))?;
    image.save_with_format("./data/out/a_out.png")?;


    let path = "./data/image/brid2.png";
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
    image.layering_mask(&result.mask, RGB(200, 0, 0))?;
    image.save_with_format("./data/out/b_out.png")?;

    Ok(())
}

fn main_image() -> Result<()> {
    let sam2 = SAM2ImageInferenceSession::new("./data/model")?;

    let path = "./data/image/brid1.png";
    let mut image = Image::open_file(path)?;

    let result = sam2.inference_sam(vec![Point { x: 92, y: 983 }], &mut image)?;

    let mut image = Image::open_file(path)?;
    let filter = AVFilter::builder(image.pixel_format()?, image.get_size())?
        .add_context("format", "rgb24")?
        .build()?;

    image.apply_filter(&filter)?;
    image.layering_mask(&result, RGB(200, 0, 0))?;
    image.save_with_format("./data/out/a_out.png")?;

    Ok(())
}