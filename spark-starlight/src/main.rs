#![feature(let_chains)]
#![cfg_attr(debug_assertions, allow(warnings))]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Result;
use log::info;
use spark_inference::disable_ffmpeg_logging;
use spark_inference::inference::sam::image_inference::{
    SAMImageInferenceSession, SamImageInference,
};
use spark_inference::inference::yolo::inference_yolo_detect::{
    YoloDetectInference, YoloDetectResult, YoloDetectSession,
};
use spark_inference::inference::yolo::NMSImplement;
use spark_inference::utils::graph::SamPrompt;
use spark_inference::utils::masks::ApplyMask;
use spark_media::filter::filter::AVFilter;
use spark_media::{Image, RGB};
use tklog::{Format, LEVEL, LOG};

fn log_init() {
    LOG.set_console(true) // Enables console logging
        .set_level(LEVEL::Info) // Sets the log level; default is Debug
        // .set_format(Format::LevelFlag | Format::Time | Format::ShortFileName)  // Defines structured log output with chosen details
        // .set_cutmode_by_size("tklogsize.txt", 1<<20, 10, true)  // Cuts logs by file size (1 MB), keeps 10 backups, compresses backups
        .uselog(); // Customizes log output format; default is "{level}{time} {file}:{message}"
}

fn main() -> Result<()> {
    log_init();
    disable_ffmpeg_logging();

    let yolo = YoloDetectSession::new("./data/model")?;
    let sam2 = SAMImageInferenceSession::new("./data/model/other4")?;

    let path = "./data/image/d4.jpg";
    let image = Image::open_file(path)?;
    let (image_width, image_height) = image.get_size();

    let results = yolo.inference_yolo(image, 0.25)?;
    info!("detect results: {:?}", results.len());

    let result_highway = results
        .clone()
        .into_iter()
        .filter(|result| result.score[0] >= 0.8)
        .collect::<Vec<_>>();
    let result_sidewalk = results
        .into_iter()
        .filter(|result| result.score[1] >= 0.4)
        .collect::<Vec<_>>();
    let result_highway = result_highway.non_maximum_suppression(0.5, 0.35, 0);
    let result_sidewalk = result_sidewalk.non_maximum_suppression(0.5, 0.25, 1);

    info!("yolo highway result: {:?}", result_highway);
    info!("yolo sidewalk result: {:?}", result_sidewalk);

    // Add this function
    fn describe_detection(
        class: &str,
        result_x: f32,
        result_y: f32,
        width: f32,
        height: f32,
    ) -> String {
        let bin_x = ((result_x / width) * 8.0).floor() as usize;
        let bin_x = bin_x.min(7);
        let direction = match bin_x {
            0 => "9 o'clock",
            1 => "10 o'clock",
            2 => "11 o'clock",
            3 => "12 o'clock",
            4 => "1 o'clock",
            5 => "2 o'clock",
            6 => "3 o'clock",
            7 => "3 o'clock",
            _ => unreachable!(),
        };
        let bin_y = ((result_y / height) * 4.0).floor() as usize;
        let bin_y = bin_y.min(3);
        let distance = match bin_y {
            0 => "far away",
            1 => "near",
            2 => "relatively close",
            3 => "very close",
            _ => unreachable!(),
        };
        format!("There is a {} at {} and {}.", class, direction, distance)
    }

    for result in &result_highway {
        let description = describe_detection(
            "highway",
            result.x,
            result.y,
            image_width as f32,
            image_height as f32,
        );
        println!("{}", description);
    }
    for result in &result_sidewalk {
        let description = describe_detection(
            "sidewalk",
            result.x,
            result.y,
            image_width as f32,
            image_height as f32,
        );
        println!("{}", description);
    }
    /*
        let image = Image::open_file(path)?;
        let result = sam2.encode_image(image)?;

        let block = get_pixel_clown();
        for (index, value) in block.iter().enumerate() {
            if value.contains(&(result_sidewalk[0].x as u32, result_sidewalk[0].y as u32)) {
                println!("sidewalk clown index: {:?}", index);
            }
        }
        let x = get_pixel_clown()
            .iter()
            .enumerate()
            .filter(|(index, value)| {
                value.contains(&(result_sidewalk[0].x as u32, result_sidewalk[0].y as u32))
            })
            .map(|(index, _)| index)
            .zip(
                get_pixel_row()
                    .iter()
                    .enumerate()
                    .filter(|(index, value)| {
                        value.contains(&(result_sidewalk[0].x as u32, result_sidewalk[0].y as u32))
                    })
                    .map(|(index, _)| index),
            )
            .collect::<Vec<_>>();

        let block = get_pixel_row();
        for (index, value) in block.iter().enumerate() {
            if value.contains(&(result_sidewalk[0].x as u32, result_sidewalk[0].y as u32)) {
                println!("sidewalk row index: {:?}", index);
            }
        }

        let highway_mask = result_highway
            .into_iter()
            .map(|yolo| {
                sam2.inference_frame(
                    SamPrompt::both(
                        (
                            yolo.x - yolo.width / 2.0,
                            yolo.y - yolo.height / 2.0,
                            yolo.x + yolo.width / 2.0,
                            yolo.y + yolo.height / 2.0,
                        ),
                        (yolo.x, yolo.y),
                    ),
                    None,
                    &result,
                )
            })
            .collect::<Vec<_>>();

        let sidewalk_mask = result_sidewalk
            .into_iter()
            .map(|yolo| {
                sam2.inference_frame(
                    SamPrompt::both(
                        (
                            yolo.x - yolo.width / 2.0,
                            yolo.y - yolo.height / 2.0,
                            yolo.x + yolo.width / 2.0,
                            yolo.y + yolo.height / 2.0,
                        ),
                        (yolo.x, yolo.y),
                    ),
                    None,
                    &result,
                )
            })
            .collect::<Vec<_>>();
    */
    Ok(())
}

fn get_pixel_row() -> Vec<Vec<(u32, u32)>> {
    let width = 640;
    let height = 640;
    let parts = 5;
    let part_height = (height as f32 / parts as f32).floor() as usize;

    let mut result = vec![Vec::new(); parts];

    for y in 0..height {
        for x in 0..width {
            let part_index = (y / part_height).min(parts - 1);
            result[part_index].push((x as u32, y as u32));
        }
    }

    result
}

fn get_pixel_clown() -> Vec<Vec<(u32, u32)>> {
    let width = 640;
    let height = 640;
    let parts = 5;
    let part_width = (width as f32 / parts as f32).floor() as usize; // 每份的宽度

    let mut result = vec![Vec::new(); parts];

    for y in 0..height {
        for x in 0..width {
            let part_index = (x / part_width).min(parts - 1); // 确定当前像素属于哪个部分
            result[part_index].push((x as u32, y as u32));
        }
    }

    result
}

fn debug() -> Result<()> {
    log_init();
    disable_ffmpeg_logging();

    let yolo = YoloDetectSession::new("./data/model")?;
    let sam2 = SAMImageInferenceSession::new("./data/model/other4")?;

    let path = "./data/image/d4.jpg";
    let image = Image::open_file(path)?;

    let results = yolo.inference_yolo(image, 0.25)?;
    println!("results: {:?}", results.len());

    let result_highway = results
        .clone()
        .into_iter()
        .filter(|result| result.score[0] >= 0.8)
        .collect::<Vec<_>>();
    let result_sidewalk = results
        .into_iter()
        .filter(|result| result.score[1] >= 0.4)
        .collect::<Vec<_>>();
    let result_highway = result_highway.non_maximum_suppression(0.5, 0.35, 0);
    let result_sidewalk = result_sidewalk.non_maximum_suppression(0.5, 0.25, 1);
    println!("highway: {:?}", result_highway);
    println!("sidewalk: {:?}", result_sidewalk);

    let image = Image::open_file(path)?;
    let result = sam2.encode_image(image)?;

    let mut image = Image::open_file(path)?;
    let (image_w, image_h) = image.get_size();
    let mut filter = AVFilter::builder(image.pixel_format()?, image.get_size())?
        .add_context("scale", "1024:1024")?
        .add_context("format", "rgb24")?;

    for mask in result_highway.iter() {
        let string1 = format!(
            "x=({x}-{width}/2):y=({y}-{height}/2):w={width}:h={height}:color=red@1.0:t=6",
            x = mask.x / image_w as f32 * 1024.0,
            y = mask.y / image_h as f32 * 1024.0,
            width = mask.width / image_w as f32 * 1024.0,
            height = mask.height / image_h as f32 * 1024.0,
        );
        filter = filter.add_context("drawbox", string1.as_str())?
    }
    for x in result_sidewalk.iter() {
        let string = format!(
            "x=({x}-{width}/2):y=({y}-{height}/2):w={width}:h={height}:color=blue@1.0:t=6",
            x = x.x / image_w as f32 * 1024.0,
            y = x.y / image_h as f32 * 1024.0,
            width = x.width / image_w as f32 * 1024.0,
            height = x.height / image_h as f32 * 1024.0,
        );
        filter = filter.add_context("drawbox", string.as_str())?
    }
    image.apply_filter(&filter.build()?)?;

    let highway_mask = result_highway
        .into_iter()
        .map(|yolo| {
            sam2.inference_frame(
                SamPrompt::both(
                    (
                        yolo.x - yolo.width / 2.0,
                        yolo.y - yolo.height / 2.0,
                        yolo.x + yolo.width / 2.0,
                        yolo.y + yolo.height / 2.0,
                    ),
                    (yolo.x, yolo.y),
                ),
                Some((1024, 1024)),
                &result,
            )
        })
        .collect::<Vec<_>>();

    let sidewalk_mask = result_sidewalk
        .into_iter()
        .map(|yolo| {
            sam2.inference_frame(
                SamPrompt::both(
                    (
                        yolo.x - yolo.width / 2.0,
                        yolo.y - yolo.height / 2.0,
                        yolo.x + yolo.width / 2.0,
                        yolo.y + yolo.height / 2.0,
                    ),
                    (yolo.x, yolo.y),
                ),
                Some((1024, 1024)),
                &result,
            )
        })
        .collect::<Vec<_>>();

    for x in highway_mask {
        if let Ok(mask) = x {
            image.layering_mask(&mask, RGB(75, 0, 0))?;
        }
    }
    for x in sidewalk_mask {
        if let Ok(mask) = x {
            image.layering_mask(&mask, RGB(0, 0, 75))?;
        }
    }
    image.save_with_format("./data/out/a_out.png")?;

    Ok(())
}
