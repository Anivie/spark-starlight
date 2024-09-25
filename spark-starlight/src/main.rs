#![cfg_attr(debug_assertions, allow(warnings))]
extern crate core;

use spark_inference::engine::inference_engine::InferenceEngine;
use anyhow::Result;
use ndarray::parallel::prelude::*;
use ndarray::s;
use spark_inference::engine::run::{InferenceResult, ModelInference};
use spark_media::Image;
use spark_media::image::decoder::size::ResizeImage;
use spark_media::image::util::extract::ExtraToTensor;

fn main() -> Result<()> {
    let mut image = {
        let mut image = Image::open_file("/home/spark-starlight/data/image/a.png")?;
        image.resize_to((640, 640))?;
        image
    };

    let mut nmg = Image::new_with_empty(image.get_size(), image.pixel_format(), image.codec_id())?;
    nmg.replace_data(image.raw_data().as_slice())?;
    nmg.save("./data/out/n.png")?;

    Ok(())
}

fn mains() -> Result<()> {
    let engine = InferenceEngine::new("./data/model/best.onnx")?;
    let mut image = {
        let mut image = Image::open_file("/home/spark-starlight/data/image/a.png")?;
        image.resize_to((640, 640))?;
        image
    };
    image.save("./data/out/a.png")?;

    let tensor = image.extra_standard_image_to_tensor()?;
    let mask = engine.inference(tensor.as_slice(), 0.8, 0.6)?;

    for (index, InferenceResult { boxes, classify, mask, score }) in mask.iter().enumerate() {
        // let frame = image.resize((640, 640), AvPixFmtRgb24)?;
        // frame.layering_mask(0, mask)?;
        // let mut image = Image::from_data((640, 640), AvPixFmtRgb24, 61)?;
        // let packet = image.fill_data(frame.get_raw_data(0).as_mut_slice())?;
        println!("Index: {}, Boxes: {:?}, Classify: {:?}, Mask: {:?}, Score: {:?}", index, boxes, classify, mask, score);

        let (mask_height, mask_width, region_height, region_width) = {
            (mask.shape()[0], mask.shape()[1], mask.shape()[0] / 3, mask.shape()[1] / 3)
        };

        for i in 0..3 {
            for j in 0..3 {
                let start_row = i * region_height;
                let end_row = if i == 2 { mask_height } else { (i + 1) * region_height };
                let start_col = j * region_width;
                let end_col = if j == 2 { mask_width } else { (j + 1) * region_width };

                let region = mask.slice(s![start_row..end_row, start_col..end_col]);
                let covered_pixels = region.iter().filter(|&&pixel| pixel > 0f32).count();
                let total_pixels = region.len();
                let coverage = covered_pixels as f64 / total_pixels as f64;

                println!("Region ({}, {}): Coverage = {:.2}%", i, j, coverage * 100.0);
            }
        }
    }

    Ok(())
}
