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
    println!("\nSidewalk_mask Guidance:");
    println!("{}", guidance);
    let guidance = analyze_road_mask(highway_mask, mask_analysis_width, mask_analysis_height);
    println!("\nHighway_mask Guidance:");
    println!("{}", guidance);
    // --- End Mask Analysis ---

    Ok(())
}

fn get_distance(normalized_y: f32) -> String {
    // normalized_y is 0.0 (top/far) to 1.0 (bottom/near)
    if normalized_y > 0.8 {
        "very near".to_string()
    } else if normalized_y > 0.5 {
        "near".to_string()
    } else if normalized_y > 0.2 {
        "mid-distance".to_string()
    } else {
        "far".to_string()
    }
}

fn get_direction(normalized_x: f32) -> String {
    // normalized_x is 0.0 (left) to 1.0 (right)
    if normalized_x < 0.3 {
        "to your left".to_string()
    } else if normalized_x > 0.7 {
        "to your right".to_string()
    } else {
        "ahead".to_string()
    }
}

fn analyze_road_mask(
    mask_results: Vec<Result<BitVec>>,
    image_width: usize,
    image_height: usize,
) -> String {
    // 1. Combine all valid masks using OR logic (Same as before)
    let combined_mask = {
        let mut combined_mask = bitvec![0; image_width * image_height];
        let mut valid_mask_count = 0;

        for result in mask_results {
            match result {
                Ok(mask) => {
                    if mask.len() == image_width * image_height {
                        combined_mask |= mask;
                        valid_mask_count += 1;
                    } else {
                        info!(
                            "Warning: Mask size mismatch. Expected {}, got {}",
                            image_width * image_height,
                            mask.len()
                        );
                    }
                }
                Err(e) => {
                    error!("Error processing mask: {:?}", e);
                }
            }
        }

        if valid_mask_count == 0 || combined_mask.not_any() {
            return "No clear sidewalk path detected.".to_string();
        }
        combined_mask
    };

    let path_at_feet = {
        // --- Add Center Check ---
        let center_x = image_width / 2;
        let center_y = image_height / 2;
        let center_index = center_y * image_width + center_x;
        // Check if the center pixel index is within the mask bounds and if it's set to true
        if center_index < combined_mask.len() {
            combined_mask[center_index]
        } else {
            false // Should not happen if mask size is correct, treat as no path if out of bounds
        }
        // --- End Center Check ---
    };

    // 2. Analyze Shape: Calculate Section Centroids and Detect Obstacles
    let num_sections = 5;
    let section_height = image_height as f64 / num_sections as f64; // Use f64 for precision
    let mut centroids_x = vec![0.0; num_sections];
    let mut pixel_counts = vec![0; num_sections];
    let mut obstacles: Vec<(usize, usize)> = Vec::new();
    let gap_threshold = (image_width as f32 * 0.1) as usize;

    for s in 0..num_sections {
        let y_start = (s as f64 * section_height).round() as usize;
        let y_end = (((s + 1) as f64 * section_height).round() as usize).min(image_height);
        let mut section_sum_x: u64 = 0;
        let mut section_count: u64 = 0;

        for y in y_start..y_end {
            let mut row_min_x = image_width;
            let mut row_max_x = 0;
            let mut current_gap_start: Option<usize> = None;
            let mut row_has_pixels = false;

            for x in 0..image_width {
                let index = y * image_width + x;
                // Check bounds just in case, though combined_mask should have the right size
                if index < combined_mask.len() && combined_mask[index] {
                    row_has_pixels = true;
                    section_sum_x += x as u64;
                    section_count += 1;
                    row_min_x = row_min_x.min(x);
                    row_max_x = row_max_x.max(x);

                    if let Some(start_x) = current_gap_start {
                        let gap_size = x - start_x;
                        if gap_size >= gap_threshold {
                            obstacles.push((y, start_x + gap_size / 2));
                        }
                        current_gap_start = None;
                    }
                } else if row_has_pixels && x > row_min_x && current_gap_start.is_none() {
                    // Only start tracking gap *after* finding the first pixel in the row
                    // and if we are not already tracking a gap.
                    if index < combined_mask.len() && !combined_mask[index] {
                        current_gap_start = Some(x);
                    }
                }
            }
            // Check for gap extending to the right edge *within the detected path*
            if let Some(start_x) = current_gap_start {
                // Ensure the gap started within or at the edge of the detected path
                if start_x <= row_max_x {
                    let gap_end = row_max_x + 1; // Consider the gap extending up to the pixel *after* the last detected one
                    let gap_size = gap_end.saturating_sub(start_x);
                    if gap_size >= gap_threshold {
                        obstacles.push((y, start_x + gap_size / 2));
                    }
                }
            }
        }

        if section_count > 0 {
            centroids_x[s] = section_sum_x as f64 / section_count as f64; // Use f64
            pixel_counts[s] = section_count as usize;
        } else {
            centroids_x[s] = -1.0; // Sentinel value indicates no path pixels in this section
        }
    }

    // Filter out sections with too few pixels
    // Use f64 for calculation, compare usize
    let min_pixels_per_section = ((image_width as f64 * section_height) / 100.0).round() as usize;
    let valid_points: Vec<(f64, f64)> = centroids_x // Vec<(y_center, x_centroid)>
        .iter()
        .enumerate()
        .filter(|(i, &cx)| cx >= 0.0 && pixel_counts[*i] > min_pixels_per_section)
        .map(|(i, &cx)| {
            // Calculate the y-center of the section
            let y_center = (i as f64 + 0.5) * section_height;
            (y_center, cx) // Store as (y, x) pair for regression
        })
        .collect();

    // 3. Determine Direction using Linear Regression on Valid Points
    let direction_str = if valid_points.len() < 2 {
        // Not enough data points for regression
        None
    } else {
        // Perform linear regression: x = m*y + c
        // Calculate sums needed for slope m = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(y^2) - (sum(y))^2)
        // Where our 'x' is centroid_x and 'y' is y_center
        let n = valid_points.len() as f64;
        let sum_y: f64 = valid_points.iter().map(|(y, _x)| y).sum();
        let sum_x: f64 = valid_points.iter().map(|(_y, x)| x).sum();
        let sum_xy: f64 = valid_points.iter().map(|(y, x)| y * x).sum();
        let sum_y2: f64 = valid_points.iter().map(|(y, _x)| y * y).sum();

        let denominator = n * sum_y2 - sum_y * sum_y;

        if denominator.abs() < 1e-6 {
            // Avoid division by zero/very small numbers (happens if all y_centers are the same)
            // Fallback to comparing the first and last point's x value if y values are identical
            if let (Some((_, first_x)), Some((_, last_x))) =
                (valid_points.first(), valid_points.last())
            {
                let centroid_shift = last_x - first_x;
                let normalized_shift = centroid_shift / image_width as f64;
                if normalized_shift.abs() < 0.05 {
                    // Threshold for straightness
                    Some("straight ahead")
                } else if normalized_shift > 0.0 {
                    // Last x > First x (moves right as y increases/gets closer) -> Veers Left looking forward
                    Some("veering slightly left")
                } else {
                    // Last x < First x (moves left as y increases/gets closer) -> Veers Right looking forward
                    Some("veering slightly right")
                }
            } else {
                // Should not happen if valid_points.len() >= 2, but handle defensively
                None
            }
        } else {
            let slope_m = (n * sum_xy - sum_x * sum_y) / denominator;

            // Define thresholds for slope interpretation (these might need tuning)
            // Slope m > 0 means x increases as y increases (path comes from left) -> Veers Left looking forward
            // Slope m < 0 means x decreases as y increases (path comes from right) -> Veers Right looking forward
            let straight_threshold = 0.1; // Example: change in x is less than 10% of change in y

            if slope_m.abs() < straight_threshold {
                Some("straight ahead")
            } else if slope_m > 0.0 {
                // Path trends left into the distance
                Some("veering slightly left") // Could add more categories like "sharply left"
            } else {
                // Path trends right into the distance
                Some("veering slightly right") // Could add "sharply right"
            }
        }
    };

    // 4. Format Output (Similar to before, but using the new direction_str)
    let base_guidance = match direction_str {
        None => {
            // If direction is unclear, still report the nearest obstacle if found
            if let Some((y, x)) = obstacles.iter().max_by_key(|(y, _)| y) {
                let obs_dist = get_distance((*y as f32 + 0.5) / image_height as f32);
                let obs_dir = get_direction((*x as f32 + 0.5) / image_width as f32);
                format!(
                    "Sidewalk path unclear, potential obstacle {} at {}",
                    obs_dist, obs_dir
                )
            } else {
                "Sidewalk path unclear.".to_string()
            }
        }
        Some(direction_str) => {
            if obstacles.is_empty() {
                format!(
                    "The sidewalk path goes {}, no immediate obstacles detected.",
                    direction_str
                )
            } else {
                // Report the nearest obstacle (largest y value)
                let nearest_obstacle = obstacles.iter().max_by_key(|(y, _)| y).unwrap(); // Safe unwrap because !obstacles.is_empty()
                let (obs_y, obs_x) = nearest_obstacle;

                // Normalize coordinates for helper functions
                let obs_dist_norm = (*obs_y as f32 + 0.5) / image_height as f32;
                let obs_dir_norm = (*obs_x as f32 + 0.5) / image_width as f32;

                let obs_dist = get_distance(obs_dist_norm);
                let obs_dir = get_direction(obs_dir_norm);

                format!(
                    "The sidewalk path goes {}, watch out for an obstacle {} at {}",
                    direction_str, obs_dist, obs_dir
                )
            }
        }
    };

    // Prepend the warning if no path was detected at the very bottom
    if !path_at_feet {
        // We already handled the case where *no* path was detected at all earlier.
        // This warning applies when a path exists but doesn't reach the bottom.
        format!(
            "Warning: No path detected immediately underfoot. {}",
            base_guidance
        )
    } else {
        base_guidance // Return the standard guidance if path reaches the bottom
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
