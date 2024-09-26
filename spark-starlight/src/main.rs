#![cfg_attr(debug_assertions, allow(warnings))]
extern crate core;

use anyhow::Result;
use spark_inference::engine::inference_engine::InferenceEngine;
use spark_inference::engine::run::{InferenceResult, ModelInference};
use spark_inference::utils::extractor::ExtraToTensor;
use spark_inference::utils::masks::ApplyMask;
use spark_media::image::decoder::size::ResizeImage;
use spark_media::{Image, RGB};

fn main() -> Result<()> {
    let engine = InferenceEngine::new("./data/model/best.onnx")?;
    let image = {
        let mut image = Image::open_file("/home/spark-starlight/data/image/a.png")?;
        image.resize_to((640, 640))?;
        image
    };

    let tensor = image.extra_standard_image_to_tensor()?;
    let mask = engine.inference(tensor.as_slice(), 0.5, 0.6)?;

    for (index, InferenceResult { boxed: boxes, classify, mask, score }) in mask.iter().enumerate() {
        println!("Index: {}, Boxes: {:?}, Classify: {:?}, Mask: {:?}, Score: {:?}", index, boxes, classify, mask.len(), score);

        let mut n_img = image.clone();
        n_img.layering_mask(0, &mask, RGB(75, 0, 0))?;

        let (mask_width, region_height, region_width) = {
            (640, mask.len() / 640 / 3, mask.len() / 640 / 3)
        };

        for i in 0..3 {
            for j in 0..3 {
                let (start_index, end_index) = {
                    let start_index = i * region_height * mask_width + j * region_width;
                    let end_index = start_index + region_height * mask_width + region_width;
                    (start_index, end_index)
                };

                let region = &mask[start_index..end_index];
                let covered_pixels = region
                    .iter()
                    .filter(|pixel| *pixel.as_ref())
                    .count();
                let coverage = covered_pixels as f64 / region.len() as f64;

                println!("Region ({}, {}): Coverage = {:.2}%", i, j, coverage * 100.0);
            }
        }
    }

    Ok(())
}
