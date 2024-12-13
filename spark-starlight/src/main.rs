#![feature(let_chains)]
#![cfg_attr(debug_assertions, allow(warnings))]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Result;
use spark_inference::inference::sam::image_inference::{SAM2ImageInferenceSession, SamImageInference};
use spark_inference::inference::yolo::inference_yolo_detect::{YoloDetectInference, YoloDetectSession};
use spark_inference::utils::graph::Point;
use spark_inference::utils::masks::ApplyMask;
use spark_media::filter::filter::AVFilter;
use spark_media::{Image, RGB};

fn main() -> Result<()> {
    let yolo = YoloDetectSession::new("./data/model")?;
    let sam2 = SAM2ImageInferenceSession::new("/home/git/SAM2Export-origin/checkpoints/tiny")?;

    let path = "./data/image/s1.jpeg";
    let image = Image::open_file(path)?;

    let results = yolo.inference_yolo(image, 0.75)?;
    println!("results: {:?}", results);

    let result_highway = results.iter().filter(|result| result.score.0 == 0).collect::<Vec<_>>();
    let result_sidewalk = results.iter().filter(|result| result.score.0 == 1).collect::<Vec<_>>();

    let image = Image::open_file(path)?;
    let result = sam2.encode_image(image)?;

    let highway_mask = sam2.decode_image(
        result_highway.iter().map(|result| Point { x: result.x, y: result.y }).collect(),
        &result,
    )?;

    let mut image = Image::open_file(path)?;
    let filter = AVFilter::builder(image.pixel_format()?, image.get_size())?
        .add_context("format", "rgb24")?
        .build()?;

    image.apply_filter(&filter)?;
    image.layering_mask(&highway_mask, RGB(200, 0, 0))?;
    image.save_with_format("./data/out/a_out.png")?;

    let sidewalk_mask = sam2.decode_image(
        result_sidewalk.iter().map(|result| Point { x: result.x, y: result.y }).collect(),
        &result,
    )?;

    let mut image = Image::open_file(path)?;
    let filter = AVFilter::builder(image.pixel_format()?, image.get_size())?
        .add_context("format", "rgb24")?
        .build()?;

    image.apply_filter(&filter)?;
    image.layering_mask(&sidewalk_mask, RGB(200, 0, 0))?;
    image.save_with_format("./data/out/b_out.png")?;


    Ok(())
}