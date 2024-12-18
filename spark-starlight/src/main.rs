#![feature(let_chains)]
#![cfg_attr(debug_assertions, allow(warnings))]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Result;
use spark_inference::inference::sam::image_inference::{
    SAM2ImageInferenceSession, SamImageInference,
};
use spark_inference::inference::yolo::inference_yolo_detect::{
    YoloDetectInference, YoloDetectSession,
};
use spark_inference::inference::yolo::NMSImplement;
use spark_inference::utils::graph::SamPrompt;
use spark_inference::utils::masks::ApplyMask;
use spark_media::filter::filter::AVFilter;
use spark_media::{Image, RGB};

fn main() -> Result<()> {
    let yolo = YoloDetectSession::new("./data/model")?;
    let sam2 = SAM2ImageInferenceSession::new("./data/model/other")?;

    let path = "./data/image/c.jpg";
    let image = Image::open_file(path)?;

    let results = yolo.inference_yolo(image, 0.3)?;
    println!("results: {:?}", results.len());

    let result_highway = results.clone().non_maximum_suppression(0.5, 0.2, 0);
    let result_sidewalk = results.non_maximum_suppression(0.5, 0.2, 1);
    println!("highway: {:?}", result_highway);
    println!("sidewalk: {:?}", result_sidewalk);

    let image = Image::open_file(path)?;
    let result = sam2.encode_image(image)?;

    let highway_mask = result_highway
        .iter()
        .map(|result| {
            SamPrompt::both(
                (result.x, result.y, result.width, result.height),
                (result.x, result.y),
            )
        })
        .map(|x| sam2.decode_image(x, &result))
        .collect::<Vec<_>>();

    let sidewalk_mask = result_sidewalk
        .iter()
        .map(|result| {
            SamPrompt::both(
                (result.x, result.y, result.width, result.height),
                (result.x, result.y),
            )
        })
        .map(|x| sam2.decode_image(x, &result))
        .collect::<Vec<_>>();

    let mut image = Image::open_file(path)?;
    let mut filter = AVFilter::builder(image.pixel_format()?, image.get_size())?
        .add_context("format", "rgb24")?;

    for x in result_highway.iter() {
        let string1 = format!(
            "x=({x}-{width}/2):y=({y}-{height}/2):w={width}:h={height}:color=red@1.0:t=2",
            x = x.x,
            y = x.y,
            width = x.width,
            height = x.height,
        );
        filter = filter.add_context("drawbox", string1.as_str())?
    }
    for x in result_sidewalk.iter() {
        let string = format!(
            "x=({x}-{width}/2):y=({y}-{height}/2):w={width}:h={height}:color=blue@1.0:t=2",
            x = x.x,
            y = x.y,
            width = x.width,
            height = x.height,
        );
        filter = filter.add_context("drawbox", string.as_str())?
    }

    image.apply_filter(&filter.build().unwrap())?;
    for x in sidewalk_mask {
        image.layering_mask(&x?, RGB(125, 0, 0))?;
    }
    for x in highway_mask {
        image.layering_mask(&x?, RGB(0, 125, 0))?;
    }
    image.save_with_format("./data/out/a_out.png")?;

    Ok(())
}
