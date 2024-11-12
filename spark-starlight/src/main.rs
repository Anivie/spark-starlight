#![feature(let_chains)]
#![cfg_attr(debug_assertions, allow(warnings))]

use anyhow::Result;
use spark_inference::engine::inference_engine::OnnxSession;
use spark_inference::inference::inference_sam::{SAM2InferenceSession, SamModelInference};
use spark_inference::init_inference_engine;
use spark_inference::utils::graph::Point;
use spark_inference::utils::masks::ApplyMask;
use spark_media::{Image, RGB};
use spark_media::filter::filter::AVFilter;

fn maind() -> Result<()> {
    use ndarray::prelude::*;

    fn rgb_to_channels(rgb_data: Array1<u8>) -> Array2<u8> {
        let n = rgb_data.len() / 3;  // 计算RGB组的数量
        rgb_data.to_shape((n, 3)).unwrap().t().to_owned()
    }

    let rgb_data = array![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

    let channels = rgb_to_channels(rgb_data);

    println!("{:?}", channels);
    println!("{:?}", concat!(file!(), ":", "name"));

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