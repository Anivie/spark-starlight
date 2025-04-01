#![feature(let_chains)]
#![cfg_attr(debug_assertions, allow(warnings))]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Result;
use bitvec::bitvec;
use bitvec::prelude::BitVec;
use log::{error, info};
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
    let (image_width, image_height) = (image_width as f32, image_height as f32);

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

    // Process highway detections
    for result in &result_highway {
        let description = describe_position(result, "highway", image_width, image_height);
        println!("{}", description);
    }

    // Process sidewalk detections
    for result in &result_sidewalk {
        let description = describe_position(result, "sidewalk", image_width, image_height);
        println!("{}", description);
    }

    let image = Image::open_file(path)?;
    let result = sam2.encode_image(image)?;

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

    // --- Analyze Sidewalk Mask ---
    // IMPORTANT: Ensure the mask size (e.g., 1024x1024) matches the width/height passed here.
    // If SAM outputs masks at a different resolution (like 1024x1024) than the original image,
    // you MUST either:
    // 1. Resize the masks to match the original image dimensions OR
    // 2. Pass the mask dimensions (1024, 1024) to analyze_road_mask instead of image_width, image_height.
    // Let's assume for now the masks are generated matching the *original* image size or resized appropriately before this step.
    // If using Some((1024, 1024)) in inference_frame, use 1024, 1024 here.
    let mask_analysis_width = 1024; // Use the actual width of the masks from SAM
    let mask_analysis_height = 1024; // Use the actual height of the masks from SAM

    let guidance = analyze_road_mask(sidewalk_mask, mask_analysis_width, mask_analysis_height);
    println!("\nNavigation Guidance:");
    println!("{}", guidance);
    let guidance = analyze_road_mask(highway_mask, mask_analysis_width, mask_analysis_height);
    println!("\nNavigation Guidance:");
    println!("{}", guidance);
    // --- End Mask Analysis ---

    Ok(())
}

fn describe_position(result: &YoloDetectResult, class: &str, width: f32, height: f32) -> String {
    // Compute center x for direction
    let x_center = result.x + result.width / 2.0;
    // Compute bottom y for distance
    let y_bottom = result.y + result.height;
    // Normalize coordinates
    let x_norm = x_center / width;
    let y_norm = y_bottom / height;
    // Get direction and distance
    let direction = get_direction(x_norm);
    let distance = get_distance(y_norm);
    // Special phrasing for "very close"
    if distance == "very close" {
        format!("the {} is under your feet at {}", class, direction)
    } else {
        format!("the {} is {} at {}", class, distance, direction)
    }
}

fn get_distance(y_norm: f32) -> String {
    if y_norm > 0.8 {
        "very close"
    } else if y_norm > 0.6 {
        "relatively close"
    } else if y_norm > 0.4 {
        "near"
    } else {
        "far"
    }
    .to_string()
}

fn get_direction(x_norm: f32) -> String {
    // Map x_norm (0 to 1) to hours from 9 to 15 (where 15 ≡ 3 o'clock)
    let h_raw = 9.0 + x_norm * 6.0;
    // Adjust hours > 12 to clock range (13 → 1, 14 → 2, 15 → 3)
    let h = if h_raw > 12.0 { h_raw - 12.0 } else { h_raw };
    let hour = h.round() as i32;
    match hour {
        9 => "9 o'clock",
        10 => "10 o'clock",
        11 => "11 o'clock",
        12 => "12 o'clock",
        1 => "1 o'clock",
        2 => "2 o'clock",
        3 => "3 o'clock",
        _ => "unknown", // Fallback (shouldn’t occur with 0 ≤ x_norm ≤ 1)
    }
    .to_string()
}

/// Analyzes the combined road mask to determine direction and obstacles.
///
/// # Arguments
/// * `mask_results` - A vector of Results, each potentially containing a BitVec mask.
/// * `image_width` - The width of the image corresponding to the masks.
/// * `image_height` - The height of the image corresponding to the masks.
///
/// # Returns
/// * A string describing the road direction and any detected obstacles.
fn analyze_road_mask(
    mask_results: Vec<Result<BitVec>>,
    image_width: usize,
    image_height: usize,
) -> String {
    // 1. Combine all valid masks using OR logic
    let combined_mask = {
        let mut combined_mask = bitvec![0; image_width * image_height];
        let mut valid_mask_count = 0;

        for result in mask_results {
            match result {
                Ok(mask) => {
                    // Ensure the mask has the expected size
                    if mask.len() == image_width * image_height {
                        combined_mask |= mask;
                        valid_mask_count += 1;
                    } else {
                        // Log or handle size mismatch if necessary
                        info!(
                            "Warning: Mask size mismatch. Expected {}, got {}",
                            image_width * image_height,
                            mask.len()
                        );
                    }
                }
                Err(e) => {
                    // Log them if needed
                    error!("Error processing mask: {:?}", e);
                }
            }
        }

        if valid_mask_count == 0 || combined_mask.not_any() {
            // combined_mask.not_any() checks if all bits are false
            return "No clear sidewalk path detected.".to_string();
        }

        combined_mask
    };

    // 2. Analyze Shape for Direction and Continuity for Obstacles
    let num_sections = 5; // Divide the height into sections (e.g., near, mid-near, middle, mid-far, far)
    let section_height = image_height / num_sections;
    let mut centroids_x = vec![0.0; num_sections];
    let mut pixel_counts = vec![0; num_sections];
    let mut obstacles: Vec<(usize, usize)> = Vec::new(); // Store obstacle (y, x) coordinates

    // Define a threshold for detecting a significant gap (e.g., % of image width)
    let gap_threshold = (image_width as f32 * 0.1) as usize; // Obstacle if gap is > 10% of image width

    for s in 0..num_sections {
        let y_start = s * section_height;
        let y_end = ((s + 1) * section_height).min(image_height);
        let mut section_sum_x: u64 = 0;
        let mut section_count: u64 = 0;

        for y in y_start..y_end {
            let mut row_min_x = image_width;
            let mut row_max_x = 0;
            let mut current_gap_start: Option<usize> = None;
            let mut row_has_pixels = false;

            for x in 0..image_width {
                let index = y * image_width + x;
                if combined_mask[index] {
                    row_has_pixels = true;
                    section_sum_x += x as u64;
                    section_count += 1;
                    row_min_x = row_min_x.min(x);
                    row_max_x = row_max_x.max(x);

                    // If we were in a gap, check if it was significant
                    if let Some(start_x) = current_gap_start {
                        let gap_size = x - start_x;
                        if gap_size >= gap_threshold {
                            // Found a significant gap, record its center
                            obstacles.push((y, start_x + gap_size / 2));
                        }
                        current_gap_start = None; // Reset gap tracking
                    }
                } else {
                    // If we are within the potential road bounds for this row and encounter false
                    if row_has_pixels && x > row_min_x && current_gap_start.is_none() {
                        current_gap_start = Some(x);
                    }
                }
            }
            // Check for a gap extending to the right edge of the detected path
            if let Some(start_x) = current_gap_start {
                if row_max_x >= start_x {
                    // Ensure the gap is within the detected bounds
                    let gap_size = row_max_x + 1 - start_x; // Gap size to the end of detected area
                    if gap_size >= gap_threshold {
                        obstacles.push((y, start_x + gap_size / 2));
                    }
                }
            }
        }

        if section_count > 0 {
            centroids_x[s] = section_sum_x as f32 / section_count as f32;
            pixel_counts[s] = section_count as usize;
        } else {
            // Handle sections with no pixels - might indicate the road ends or a huge obstacle
            centroids_x[s] = -1.0; // Sentinel value
        }
    }

    // Filter out sections with too few pixels to be reliable
    let min_pixels_per_section = (image_width * section_height) / 100; // e.g., require at least 1% of section area
    let valid_centroids: Vec<(usize, f32)> = centroids_x
        .iter()
        .enumerate()
        .filter(|(i, &cx)| cx >= 0.0 && pixel_counts[*i] > min_pixels_per_section)
        .map(|(i, &cx)| (i, cx))
        .collect();

    if valid_centroids.len() < 2 {
        // Not enough data to determine direction reliably
        // Check for obstacles found even without clear direction
        return if let Some((y, x)) = obstacles.first() {
            let obs_dist = get_distance((*y as f32 + 0.5) / image_height as f32); // Use center of pixel row
            let obs_dir = get_direction((*x as f32 + 0.5) / image_width as f32); // Use center of gap
            format!(
                "Sidewalk path unclear, potential obstacle {} at {}",
                obs_dist, obs_dir
            )
        } else {
            "Sidewalk path unclear.".to_string()
        };
    }

    // Determine direction based on the first (nearest, largest index) and last (farthest, smallest index) valid centroids
    // Note: Index 0 is farthest (top), index num_sections-1 is nearest (bottom).
    // We need to reverse this logic slightly: Compare nearest valid section to farthest valid section.
    let (_near_section_idx, near_centroid_x) = valid_centroids.last().unwrap(); // Nearest section with data
    let (_far_section_idx, far_centroid_x) = valid_centroids.first().unwrap(); // Farthest section with data

    let centroid_shift = near_centroid_x - far_centroid_x;
    let normalized_shift = centroid_shift / image_width as f32; // Normalize shift by image width

    let direction_str = if normalized_shift.abs() < 0.05 {
        // Threshold for straightness (e.g., shift < 5% of width)
        "straight ahead"
    } else if normalized_shift > 0.0 {
        // Near centroid is to the right of far centroid -> path veers right from perspective
        "veers slightly right" // Or "veers right" if shift is larger
    } else {
        // Near centroid is to the left of far centroid -> path veers left
        "veers slightly left" // Or "veers left"
    };

    // 3. Format Output
    if obstacles.is_empty() {
        format!(
            "The sidewalk path goes {}, no immediate obstacles detected.",
            direction_str
        )
    } else {
        // Report the nearest obstacle first
        // Obstacles are stored (y, x). Larger y means closer.
        let nearest_obstacle = obstacles.iter().max_by_key(|(y, _)| y).unwrap();
        let (obs_y, obs_x) = nearest_obstacle;

        let obs_dist = get_distance((*obs_y as f32 + 0.5) / image_height as f32); // Use center of pixel row
        let obs_dir = get_direction((*obs_x as f32 + 0.5) / image_width as f32); // Use center of gap

        format!(
            "The sidewalk path goes {}, watch out for an obstacle {} at {}",
            direction_str, obs_dist, obs_dir
        )
    }
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
