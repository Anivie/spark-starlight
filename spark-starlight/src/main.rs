#![feature(let_chains)]
#![cfg_attr(debug_assertions, allow(warnings))]

use crate::detect::mask::{analyze_road_mask, get_best_highway};
use bitvec::prelude::BitVec;
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
use spark_media::Image;
use std::sync::Arc;
use tokio::task::{spawn_blocking, JoinHandle};

mod debug;
mod detect;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn log_init() {
    tracing_subscriber::fmt::init();
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    log_init();
    disable_ffmpeg_logging();

    let yolo = Arc::new(YoloDetectSession::new("./data/model")?);
    let sam2 = Arc::new(SAMImageInferenceSession::new("./data/model/other5")?);

    let path = "./data/image/rt.jpeg";
    let image = Image::open_file(path)?;
    let sam_image = image.clone();
    let (image_width, image_height) = (image.get_width() as u32, image.get_height() as u32);

    let results: JoinHandle<anyhow::Result<Vec<YoloDetectResult>>> =
        spawn_blocking(move || Ok(yolo.inference_yolo(image.clone(), 0.25)?));
    let results = results.await??;
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
    let mut result_highway = result_highway.non_maximum_suppression(0.5, 0.35, 0);
    let mut result_sidewalk = result_sidewalk.non_maximum_suppression(0.5, 0.25, 1);

    let mask = {
        let result_highway = result_highway
            .iter()
            .map(|yolo| {
                SamPrompt::both(
                    (
                        yolo.x - yolo.width / 2.0,
                        yolo.y - yolo.height / 2.0,
                        yolo.x + yolo.width / 2.0,
                        yolo.y + yolo.height / 2.0,
                    ),
                    (yolo.x, yolo.y),
                )
            })
            .collect::<Vec<_>>();

        let result_sidewalk = result_sidewalk
            .iter()
            .map(|yolo| {
                SamPrompt::both(
                    (
                        yolo.x - yolo.width / 2.0,
                        yolo.y - yolo.height / 2.0,
                        yolo.x + yolo.width / 2.0,
                        yolo.y + yolo.height / 2.0,
                    ),
                    (yolo.x, yolo.y),
                )
            })
            .collect::<Vec<_>>();

        let sam2 = sam2.clone();
        let handle: JoinHandle<anyhow::Result<Vec<Vec<BitVec>>>> = spawn_blocking(move || {
            Ok(sam2.inference_frame(
                sam_image,
                Some((1024, 1024)),
                vec![result_highway, result_sidewalk],
            )?)
        });

        handle.await??
    };

    // Rescale the yolo results to 1024x1024
    for yolo in result_highway.iter_mut() {
        yolo.x = yolo.x / image_width as f32 * 1024.0;
        yolo.y = yolo.y / image_height as f32 * 1024.0;
        yolo.width = yolo.width / image_width as f32 * 1024.0;
        yolo.height = yolo.height / image_height as f32 * 1024.0;
    }
    for yolo in result_sidewalk.iter_mut() {
        yolo.x = yolo.x / image_width as f32 * 1024.0;
        yolo.y = yolo.y / image_height as f32 * 1024.0;
        yolo.width = yolo.width / image_width as f32 * 1024.0;
        yolo.height = yolo.height / image_height as f32 * 1024.0;
    }

    println!("--- Analyzing Highway ---");
    // Filter highway masks: prioritize closest to user (highest average y)
    if let Some(mask) = get_best_highway(&mask[0]) {
        println!(
            "Highway: {}",
            analyze_road_mask(mask, result_highway.as_slice(), 1024, 1024, "Highway").await
        );
    }

    println!("\n--- Analyzing Sidewalk ---");
    // Filter sidewalk masks: first find those under user's feet (contains bottom center)
    let user_x = 1024 / 2;
    let user_y = 1024 - 1; // Bottom center pixel
    let mut valid_sidewalks: Vec<_> = mask[1]
        .iter()
        .filter(|mask| mask[user_y * 1024 + user_x]) // Check if contains user position
        .collect();

    // If none under feet, use all masks
    if valid_sidewalks.is_empty() {
        valid_sidewalks = mask[1].iter().collect();
    }

    // Select sidewalk with the largest area (most true bits)
    let best_sidewalk_mask = valid_sidewalks.iter().max_by_key(|mask| mask.count_ones());

    if let Some(mask) = best_sidewalk_mask {
        println!(
            "Sidewalk: {}",
            analyze_road_mask(mask, result_sidewalk.as_slice(), 1024, 1024, "Sidewalk").await
        );
    }

    Ok(())
}
