#![feature(let_chains)]
#![cfg_attr(debug_assertions, allow(warnings))]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Result;
use spark_inference::disable_ffmpeg_logging;
use spark_inference::inference::sam::image_inference::{
    SAMImageInferenceSession, SamImageInference,
};
use spark_inference::inference::yolo::inference_yolo_detect::{
    YoloDetectInference, YoloDetectSession,
};
use spark_inference::inference::yolo::NMSImplement;
use spark_inference::utils::graph::SamPrompt;
use spark_media::filter::filter::AVFilter;
use spark_media::Image;

fn main() -> Result<()> {
    disable_ffmpeg_logging();

    let yolo = YoloDetectSession::new("./data/model")?;
    let sam2 = SAMImageInferenceSession::new("./data/model/other3")?;

    let path = "./data/image/rd.jpg";
    let image = Image::open_file(path)?;

    let results = yolo.inference_yolo(image, 0.25)?;
    println!("results: {:?}", results.len());

    let result_highway = results
        .clone()
        .into_iter()
        .filter(|result| result.score[0] >= 0.25)
        .collect::<Vec<_>>();
    let result_sidewalk = results
        .into_iter()
        .filter(|result| result.score[1] >= 0.25)
        .collect::<Vec<_>>();
    let result_highway = result_highway.non_maximum_suppression(0.5, 0.35, 0);
    let result_sidewalk = result_sidewalk.non_maximum_suppression(0.5, 0.25, 1);
    println!("highway: {:?}", result_highway);
    println!("sidewalk: {:?}", result_sidewalk);

    let image = Image::open_file(path)?;
    let result = sam2.encode_image(image)?;

    let mut image = Image::open_file(path)?;
    let mut filter = AVFilter::builder(image.pixel_format()?, image.get_size())?
        .add_context("format", "rgb24")?;

    for mask in result_highway.iter() {
        let string1 = format!(
            "x=({x}-{width}/2):y=({y}-{height}/2):w={width}:h={height}:color=red@1.0:t=6",
            x = mask.x,
            y = mask.y,
            width = mask.width,
            height = mask.height,
        );
        filter = filter.add_context("drawbox", string1.as_str())?
    }
    for x in result_sidewalk.iter() {
        let string = format!(
            "x=({x}-{width}/2):y=({y}-{height}/2):w={width}:h={height}:color=blue@1.0:t=6",
            x = x.x,
            y = x.y,
            width = x.width,
            height = x.height,
        );
        filter = filter.add_context("drawbox", string.as_str())?
    }
    image.apply_filter(&filter.build()?)?;

    let highway_mask = result_highway
        .into_iter()
        .map(|x| {
            sam2.inference_frame(
                SamPrompt::both(
                    (
                        x.x - x.width / 2.0,
                        x.y - x.height / 2.0,
                        x.x + x.width / 2.0,
                        x.y + x.height / 2.0,
                    ),
                    (x.width, x.height),
                ),
                &result,
            )
        })
        .collect::<Vec<_>>();

    let sidewalk_mask = result_sidewalk
        .into_iter()
        .map(|x| {
            sam2.inference_frame(
                SamPrompt::both(
                    (
                        x.x - x.width / 2.0,
                        x.y - x.height / 2.0,
                        x.x + x.width / 2.0,
                        x.y + x.height / 2.0,
                    ),
                    (x.width, x.height),
                ),
                &result,
            )
        })
        .collect::<Vec<_>>();

    for x in highway_mask {
        // image.layering_mask(&x?, RGB(0, 75, 0))?;
    }
    for x in sidewalk_mask {
        // image.layering_mask(&x?, RGB(75, 0, 0))?;
    }
    image.save_with_format("./data/out/a_out.png")?;

    Ok(())
}
