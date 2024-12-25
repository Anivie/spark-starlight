#![feature(let_chains)]
#![cfg_attr(debug_assertions, allow(warnings))]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Result;
use spark_inference::disable_ffmpeg_logging;
use spark_inference::inference::sam::video_inference::video_inference::{
    SAMVideoInferenceSession, SamVideoInference,
};
use spark_inference::inference::sam::video_inference::InferenceInput;
use spark_inference::utils::graph::SamPrompt;
use spark_inference::utils::masks::ApplyMask;
use spark_media::filter::filter::AVFilter;
use spark_media::{Image, RGB};

fn main() -> Result<()> {
    disable_ffmpeg_logging();
    let path = "./data/image/bed1.png";

    let sam2 = SAMVideoInferenceSession::new("./data/model/other3")?;
    let image = Image::open_file(path)?;
    let encoded = sam2.encode_image(image)?;
    let (mask, state) = sam2.inference_frame(
        // InferenceInput::Prompt(SamPrompt::both((624.0, 626.0, 125.0, 97.0), (675.0, 648.0))),
        // InferenceInput::Prompt(SamPrompt::point(675., 648.)),
        InferenceInput::Prompt(SamPrompt::point(210., 350.)),
        &encoded,
    )?;

    let mut image = Image::open_file(path)?;
    let filter = AVFilter::builder(image.pixel_format()?, image.get_size())?
        .add_context("scale", "1024:1024")?
        .add_context("format", "rgb24")?
        .build()?;
    image.apply_filter(&filter)?;
    image.layering_mask(&mask, RGB(0, 65, 45))?;
    image.save("./data/out/bird1_mask.png")?;

    let path = "./data/image/bed2.png";
    let image = Image::open_file(path)?;
    let encoded = sam2.encode_image(image)?;
    let (mask, state) = sam2.inference_frame(InferenceInput::State(state), &encoded)?;

    let mut image = Image::open_file(path)?;
    let filter = AVFilter::builder(image.pixel_format()?, image.get_size())?
        .add_context("scale", "1024:1024")?
        .add_context("format", "rgb24")?
        .build()?;
    image.apply_filter(&filter)?;
    image.layering_mask(&mask, RGB(64, 0, 0))?;
    image.save("./data/out/bird2_mask.png")?;

    let path = "./data/image/bed3.png";
    let image = Image::open_file(path)?;
    let encoded = sam2.encode_image(image)?;
    let (mask, _) = sam2.inference_frame(InferenceInput::State(state), &encoded)?;

    let mut image = Image::open_file(path)?;
    let filter = AVFilter::builder(image.pixel_format()?, image.get_size())?
        .add_context("scale", "1024:1024")?
        .add_context("format", "rgb24")?
        .build()?;
    image.apply_filter(&filter)?;
    image.layering_mask(&mask, RGB(64, 0, 0))?;
    image.save("./data/out/bird3_mask.png")?;

    Ok(())
}
