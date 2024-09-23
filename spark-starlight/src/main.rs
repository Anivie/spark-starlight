#![cfg_attr(debug_assertions, allow(warnings))]
extern crate core;

use spark_inference::engine::inference_engine::InferenceEngine;
use anyhow::Result;
use ndarray::parallel::prelude::*;
use ndarray::s;
use spark_inference::engine::run::{InferenceResult, ModelInference};
use spark_media::AVPixelFormat::AvPixFmtRgb24;
use spark_media::Image;
use spark_media::image::decoder::size::ResizeImage;
use spark_media::image::util::extract::ExtraToTensor;

fn main() -> Result<()> {
    let engine = InferenceEngine::new("./data/model/best.onnx")?;
    let mut image = Image::open_file("/home/spark-starlight/data/image/b.png")?;

    image.resize((640, 640), AvPixFmtRgb24)?;
    let tensor = image.extra_standard_image_to_tensor()?;
    let mask = engine.inference(tensor.as_slice(), 0.8, 0.6)?;

    for (index, InferenceResult { boxes, classify, mask, score }) in mask.iter().enumerate() {
        // let frame = image.resize((640, 640), AvPixFmtRgb24)?;
        // frame.layering_mask(0, mask)?;
        // let mut image = Image::from_data((640, 640), AvPixFmtRgb24, 61)?;
        // let packet = image.fill_data(frame.get_raw_data(0).as_mut_slice())?;

        let masks = &mask.slice(s![600.., ..]);
        //根据masks中的数据，判断是否可以行走
        let mut walk = false;
        for i in 0..masks.shape()[0] {
            for j in 0..masks.shape()[1] {
                if masks[[i, j]] > 0.5 {
                    walk = true;
                    break;
                }
            }
        }
    }

    Ok(())
}
