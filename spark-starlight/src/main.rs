#![cfg_attr(debug_assertions, allow(warnings))]

use anyhow::Result;
use spark_inference::engine::inference_engine::OnnxSession;
use spark_inference::inference::inference_sam::{SAM2InferenceSession, SamModelInference};

fn main() -> Result<()> {
    let image_encoder = OnnxSession::new("./data/model/image_encoder.onnx")?;
    let image_decoder = OnnxSession::new("./data/model/image_decoder.onnx")?;
    let memory_attention = OnnxSession::new("./data/model/memory_attention.onnx")?;
    let memory_encoder = OnnxSession::new("./data/model/memory_encoder.onnx")?;

    println!("image_encoder: {:?}, out: {:?}\n", image_encoder.inputs, image_encoder.outputs);
    println!("memory_attention: {:?}", memory_attention.inputs);
    /*println!("image_decoder: {:?}", image_decoder.inputs);

    println!("memory_encoder: {:?}", memory_encoder.inputs);*/

    let sam2 = SAM2InferenceSession::new(image_encoder, image_decoder, memory_attention, memory_encoder);

    Ok(())
}