#![feature(let_chains)]
#![cfg_attr(debug_assertions, allow(warnings))]

use crate::detect::mask::{analyze_road_mask, get_best_highway};
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
use std::clone;
use std::ops::Deref;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use tokio::task::{spawn_blocking, JoinHandle};
use tokio::{count, join, spawn};
use zenoh::sample::Sample;
use zenoh::{Config, Session};

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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    log_init();
    disable_ffmpeg_logging();

    let yolo = YoloDetectSession::new("./data/model")?;
    let sam2 = SAMImageInferenceSession::new("./data/model/other5")?;
    let tts = TTSEngine::new_en()?;
    let engine = InferenceEngine { yolo, sam2, tts };
    let engine: &'static InferenceEngine = Box::leak(Box::new(engine));

    let session = {
        let config = Config::from_json5(
            r#"
         {
             mode: "router",
             listen: {
                endpoints: ["tcp/0.0.0.0:7447"],
             },
             transport: {
                link: {
                    rx: {
                        buffer_size: 1048576
                    }
                }
             }
         }
        "#,
        )
        .unwrap();
        Arc::new(zenoh::open(config).await.unwrap())
    };
    let subscriber = session.declare_subscriber("spark/server").await.unwrap();

    while let Ok(sample) = subscriber.recv_async().await {
        let payload = sample.payload().to_bytes();
        let tag = payload.split(|&b| b == b':').next().unwrap_or(&payload);
        let session_id = &payload[tag.len() + 1..];
        let session_id = String::from_utf8_lossy(session_id).to_string();

        match tag {
            b"uploadImage" => {
                handle_upload(engine, session.clone(), sample, session_id);
            }
            _ => {
                info!(
                    "Received unknown payload: {}",
                    String::from_utf8_lossy(sample.payload().to_bytes().deref())
                );
            }
        }
    }

    Ok(())
}

fn handle_upload(
    engine: &'static InferenceEngine,
    session: Arc<Session>,
    sample: Sample,
    session_id: String,
) {
    if sample.attachment().is_none() {
        warn!("No attachment found");
        return;
    };
    info!("Got uploadImage request from client: {}", session_id);

    let session = session.clone();
    spawn(async move {
        let result: anyhow::Result<()> = async {
            let image = spawn_blocking(move || {
                let attachment = sample.attachment().unwrap();
                Image::from_bytes(attachment.to_bytes())
            })
            .await??;

            let result = analyse_image(image, engine).await?;
            let builder = session
                .put(format!("spark/client/{}", session_id), "returnImageInfo")
                .attachment(result);
            if let Err(e) = builder.await {
                warn!("Error sending response: {}", e);
            } else {
                info!("Response sent to client: {}", session_id);
            };
            Ok(())
        }
        .await;

        if let Err(e) = result {
            session
                .put(
                    format!("spark/client/{}", session_id),
                    "serverInternalError",
                )
                .await
                .unwrap();
            error!("Error processing request: {}", e);
        }
    });
}

async fn analyse_image(image: Image, engine: &'static InferenceEngine) -> anyhow::Result<Vec<u8>> {
    let (image_width, image_height) = (image.get_width() as u32, image.get_height() as u32);
    let (yolo, sam, tts) = (&engine.yolo, &engine.sam2, &engine.tts);

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

        let handle: JoinHandle<anyhow::Result<Vec<Vec<BitVec>>>> = spawn_blocking(move || {
            Ok(sam.inference_frame(
                image,
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

    let mut back = String::new();
    // Filter highway masks: prioritize closest to user (highest average y)
    if let (Some(highway), Some(sidewalk)) = (best_highway, best_sidewalk_mask) {
        let (highway, sidewalk) = join!(
            analyze_road_mask(highway, result_highway.as_slice(), 1024, 1024, "Highway"),
            analyze_road_mask(sidewalk, result_sidewalk.as_slice(), 1024, 1024, "Sidewalk"),
        );
        back.push_str(&highway);
        back.push_str(&sidewalk);
    } else if let Some(highway) = best_highway {
        let highway =
            analyze_road_mask(highway, result_highway.as_slice(), 1024, 1024, "Highway").await;
        back.push_str(&highway);
    } else if let Some(sidewalk) = best_sidewalk_mask {
        let sidewalk =
            analyze_road_mask(sidewalk, result_sidewalk.as_slice(), 1024, 1024, "Sidewalk").await;
        back.push_str(&sidewalk);
    }

    let result: anyhow::Result<Vec<u8>> = spawn_blocking(move || {
        Ok(tts.generate(if back.is_empty() {
            "Warning: No road detected"
        } else {
            &back
        })?)
    })
    .await?;

    Ok(result?)
}
