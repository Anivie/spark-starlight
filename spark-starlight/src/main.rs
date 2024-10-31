#![cfg_attr(debug_assertions, allow(warnings))]

use anyhow::Result;
use bitvec::order::Lsb0;
use bitvec::prelude::BitVec;
use ndarray::parallel::prelude::*;
use spark_inference::engine::inference_engine::InferenceEngine;
use spark_inference::engine::run::ModelInference;
use spark_inference::utils::extractor::ExtraToTensor;
use spark_inference::utils::masks::ApplyMask;
use spark_media::{Image, RGB};

fn main() -> Result<()> {
    // let mut image = Image::open_file("./data/image/s1.jpeg")?;

    let engine = InferenceEngine::new("./data/model/best2.onnx")?;
    for (index, x) in std::fs::read_dir("./data/image")?.enumerate() {
        let image = Image::open_file(x?.path().to_str().unwrap())?;
        run(image, &engine, index)?
    }

    Ok(())
}

fn run(mut image: Image, engine: &InferenceEngine, index: usize) -> Result<()> {
    image.add_filter()?;
    println!("pixel format: {:?}", image.pixel_format());

    let number_of_species = 2;

    let tensor = image.extra_standard_image_to_tensor()?;
    let mask = engine.inference(tensor.as_slice(), 0.25, 0.45)?;

    let masks: Vec<BitVec> = (0..number_of_species)
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
                RGB(0, 0, 125)
            } else {
                RGB(125, 0, 0)
            },
        )?;
    }

    image.save(format!("./data/out/{}.png", index))?;

    Ok(())
}