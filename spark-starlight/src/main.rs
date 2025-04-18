#![feature(let_chains)]
#![cfg_attr(debug_assertions, allow(warnings))]

use crate::detect::mask::{analyze_road_mask, get_best_highway};
use actix_web::{web, App, Error, HttpResponse, HttpServer};
use bitvec::prelude::BitVec;
use bytes::Bytes;
use log::{error, info, warn};
use spark_inference::disable_ffmpeg_logging;
use spark_inference::inference::sam::image_inference::{
    SAMImageInferenceSession, SamImageInference,
};
use spark_inference::inference::tts::tts_engine::{TTSEngine, TTS};
use spark_inference::inference::yolo::inference_yolo_detect::{
    YoloDetectInference, YoloDetectResult, YoloDetectSession,
};
use spark_inference::inference::yolo::NMSImplement;
use spark_inference::utils::graph::SamPrompt;
use spark_media::Image;
use std::ops::Deref;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use tokio::task::{spawn_blocking, JoinHandle};
use tokio::{count, join, spawn};

mod debug;
mod detect;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn log_init() {
    tracing_subscriber::fmt::init();
}

struct InferenceEngine {
    yolo: YoloDetectSession,
    sam2: SAMImageInferenceSession,
    tts: TTSEngine,
}

async fn upload_image_handler(
    engine: web::Data<&'static InferenceEngine>,
    body: Bytes,
) -> Result<HttpResponse, Error> {
    info!("Received POST request on /uploadImage");

    let image = match spawn_blocking(move || Image::from_bytes(body.deref())).await {
        Ok(Ok(image)) => image,
        err => {
            warn!("Error processing image: {:?}", err);
            return Ok(HttpResponse::BadRequest().body("Invalid image data"));
        }
    };

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    image
        .clone()
        .save_with_format(format!("./data/image/upload/input_{}.jpg", timestamp))
        .unwrap();

    match analyse_image(image, engine.deref()).await {
        Ok(response_message) => {
            info!("Processing successful.");
            Ok(HttpResponse::Ok().body(response_message))
        }
        Err(e) => {
            log::error!("Error processing image: {:?}", e);
            Ok(HttpResponse::InternalServerError().body(format!("Error processing image: {}", e)))
        }
    }
}

#[actix_web::main]
async fn main() -> anyhow::Result<()> {
    log_init();
    disable_ffmpeg_logging();

    let yolo = YoloDetectSession::new("./data/model")?;
    let sam2 = SAMImageInferenceSession::new("./data/model/other5")?;
    let tts = TTSEngine::new_en()?;
    let engine: &'static InferenceEngine = Box::leak(Box::new(InferenceEngine { yolo, sam2, tts }));

    HttpServer::new(move || {
        App::new()
            .app_data(web::PayloadConfig::new(1024 * 1024 * 1024 * 25))
            .app_data(web::Data::new(engine))
            .route("/uploadImage", web::post().to(upload_image_handler))
    })
    .bind(("0.0.0.0", 7447))?
    .run()
    .await?;

    Ok(())
}

async fn analyse_image(image: Image, engine: &'static InferenceEngine) -> anyhow::Result<Vec<u8>> {
    let (image_width, image_height) = (image.get_width() as u32, image.get_height() as u32);
    let (yolo, sam, tts) = (&engine.yolo, &engine.sam2, &engine.tts);

    info!("Start Yolo detection");
    let results = {
        let image = image.clone();
        spawn_blocking(move || {
            Ok::<Vec<YoloDetectResult>, anyhow::Error>(yolo.inference_yolo(image, 0.25)?)
        })
        .await??
    };

    info!("detect results: {:?}", results.len());
    if results.is_empty() {
        let result: anyhow::Result<Vec<u8>> = spawn_blocking(move || {
            Ok(tts.generate("Warning: Nothing could be found in current scene")?)
        })
        .await?;

        return Ok(result?);
    }

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
    info!(
        "Non max suppression results: {} {}",
        result_highway.len(),
        result_sidewalk.len()
    );

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

        info!("Start SAM inference");
        let handle: JoinHandle<anyhow::Result<Vec<Vec<BitVec>>>> = spawn_blocking(move || {
            Ok(sam.inference_frame(
                image,
                Some((1024, 1024)),
                vec![result_highway, result_sidewalk],
            )?)
        });

        handle.await??
    };
    info!("SAM inference results: {} {}", mask[0].len(), mask[1].len());

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

    let best_highway = get_best_highway(&mask[0]);

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

    info!("Start analyzing masks");
    let mut back = Vec::new();
    // Filter highway masks: prioritize closest to user (highest average y)
    let highway_future = best_highway
        .map(|hw| analyze_road_mask(hw, result_highway.as_slice(), 1024, 1024, "Highway"));

    let sidewalk_future = best_sidewalk_mask
        .map(|sw| analyze_road_mask(sw, result_sidewalk.as_slice(), 1024, 1024, "Sidewalk"));

    match (highway_future, sidewalk_future) {
        (Some(hw), Some(sw)) => {
            let (highway, sidewalk) = join!(hw, sw);
            back.push(highway);
            back.push(sidewalk);
        }
        (Some(hw), None) => {
            let highway = hw.await;
            back.push(highway);
        }
        (None, Some(sw)) => {
            let sidewalk = sw.await;
            back.push(sidewalk);
        }
        (None, None) => {}
    }

    let string = back.join(", ");
    info!("Get natural language: {}", string);
    let result: anyhow::Result<Vec<u8>> = spawn_blocking(move || {
        Ok(tts.generate(if back.is_empty() {
            "Warning: No road detected"
        } else {
            string.as_str()
        })?)
    })
    .await?;

    Ok(result?)
}
