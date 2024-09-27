#![cfg_attr(debug_assertions, allow(warnings))]

use anyhow::Result;
use spark_inference::engine::inference_engine::InferenceEngine;
use spark_inference::engine::run::{InferenceResult, ModelInference};
use spark_inference::utils::extractor::ExtraToTensor;
use spark_inference::utils::masks::ApplyMask;
use spark_media::image::decoder::size::ResizeImage;
use spark_media::{Image, RGB};

fn main() -> Result<()> {
    let engine = InferenceEngine::new("./data/model/best.onnx")?;
    let mut image = {
        let mut image = Image::open_file("./data/image/a.png")?;
        image.resize_to((640, 640))?;
        image
    };

    let tensor = image.extra_standard_image_to_tensor()?;
    let mask = engine.inference(tensor.as_slice(), 0.25, 0.45)?;

    for InferenceResult { boxed: boxes, classify, mask, score } in mask.iter() {
        println!("Boxes: {:?}, Classify: {:?}, Mask: {:?}, Score: {:?}", boxes, classify, mask.len(), score);
        let best = classify.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(index, _)| index)
            .unwrap();

        image.layering_mask(
            &mask,
            if best == 0 {
                RGB(0, 25, 25)
            } else {
                RGB(25, 25, 0)
            },
        )?;

        /*let (mask_width, region_height, region_width) = {
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
                    .filter(|pixel| **pixel)
                    .count();
                let coverage = covered_pixels as f64 / region.len() as f64;

                println!("Region ({}, {}): Coverage = {:.2}%", i, j, coverage * 100.0);
            }
        }*/
    }

    image.save(&format!("./data/out/{}.png", "best"))?;

    Ok(())
}
