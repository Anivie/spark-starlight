#![cfg_attr(debug_assertions, allow(warnings))]

use anyhow::Result;
use bitvec::order::Lsb0;
use bitvec::prelude::BitVec;
use ndarray::parallel::prelude::*;
use spark_inference::engine::inference_engine::InferenceEngine;
use spark_inference::engine::run::ModelInference;
use spark_inference::utils::extractor::ExtraToTensor;
use spark_inference::utils::masks::ApplyMask;
use spark_media::image::decoder::size::ResizeImage;
use spark_media::{AVFilterGraph, Image, RGB};

fn main() -> Result<()> {
    let mut image = Image::open_file("./data/image/rt.jpeg")?;
    // image.resize_to((640, 640))?;
    image.resize()?;
    image.save(&format!("./data/out/{}.png", "resize"))?;
    return Ok(());

    let engine = InferenceEngine::new("./data/model/best.onnx")?;
    let tensor = image.extra_standard_image_to_tensor()?;
    let mask = engine.inference(tensor.as_slice(), 0.25, 0.45)?;
    let masks: Vec<_> = (0..2)
        .into_par_iter()
        .map(|class_index| {
            let mut mask_all = BitVec::<usize, Lsb0>::repeat(false, 640 * 640);
            mask.iter()
                .filter(|x| x.classify == class_index)
                .flat_map(|each_class| each_class.mask.iter().enumerate())
                .for_each(|(index, all)| {
                    if *all {
                        mask_all.set(index, true);
                    }
                });
            mask_all
        })
        .collect();

    for (index, mask) in masks.iter().enumerate() {
        image.layering_mask(
            mask,
            if index == 0 {
                RGB(0, 25, 25)
            } else {
                RGB(25, 25, 0)
            },
        )?;
    }

    image.save(&format!("./data/out/{}.png", "best"))?;

    Ok(())
}
