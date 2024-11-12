#![feature(let_chains)]
#![cfg_attr(debug_assertions, allow(warnings))]

use anyhow::Result;
use spark_inference::engine::inference_engine::OnnxSession;
use spark_inference::inference::inference_sam::{SAM2InferenceSession, SamModelInference};
use spark_inference::inference::inference_yolo::YoloModelInference;
use spark_inference::init_inference_engine;
use spark_inference::utils::graph::Point;
use spark_inference::utils::masks::ApplyMask;
use spark_media::{Image, RGB};
use spark_media::filter::filter::AVFilter;

fn main_yolo() -> Result<()> {
    init_inference_engine()?;

    let path = "./data/image/a.png";
    let image = Image::open_file(path)?;

    let yolo = OnnxSession::new("./data/model/best2.onnx")?;

    let results = yolo.inference_yolo(image, 0.25, 0.45)?;
    println!("{:?}", results.len());

    let mut image = Image::open_file(path)?;
    let filter = {
        let mut filter = AVFilter::new(image.pixel_format()?, image.get_size())?;
        filter.add_context("scale", "640:640:force_original_aspect_ratio=decrease")?;
        filter.add_context("pad", "640:640:(ow-iw)/2:(oh-ih)/2:#727272")?;
        filter.add_context("format", "rgb24")?;
        filter.lock()?
    };
    image.apply_filter(&filter)?;

    for result in results {
        image.layering_mask(&result.mask, RGB(20, 20, 0))?;
    }

    image.save_with_format("./data/out/y_out.png")?;

    Ok(())
}

fn main() -> Result<()> {
    init_inference_engine()?;

    let image_encoder = OnnxSession::new("./data/model/image_encoder.onnx")?;
    let image_decoder = OnnxSession::new("./data/model/image_decoder.onnx")?;
    let memory_attention = OnnxSession::new("./data/model/memory_attention.onnx")?;
    let memory_encoder = OnnxSession::new("./data/model/memory_encoder.onnx")?;

    println!("image_encoder: {:?} ======== out: {:?}\n", image_encoder.inputs, image_encoder.outputs);
    println!("image_decoder: {:?} ======== out: {:?}\n", image_decoder.inputs, image_decoder.outputs);
    println!("memory_attention: {:?} ======== out: {:?}\n", memory_attention.inputs, memory_attention.outputs);
    println!("memory_encoder: {:?} ======== out: {:?}\n", memory_encoder.inputs, memory_encoder.outputs);

    let sam2 = SAM2InferenceSession::new(image_encoder, image_decoder, memory_attention, memory_encoder);

    let path = "./data/image/a.png";
    let mut image = Image::open_file(path)?;

    let result = sam2.inference_sam(vec![Point { x: 100, y: 20 }, Point { x: 300, y: 20 }], &mut image)?;

    let mut image = Image::open_file(path)?;
    let filter = {
        let mut filter = AVFilter::new(image.pixel_format()?, image.get_size())?;
        filter.add_context("format", "rgb24")?;
        filter.lock()?
    };
    image.apply_filter(&filter)?;
    image.layering_mask(&result, RGB(200, 0, 0))?;
    image.save_with_format("./data/out/a_out.png")?;

    Ok(())
}