#![feature(let_chains)]
#![cfg_attr(debug_assertions, allow(warnings))]

mod debug;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Result;
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
use std::f32::consts::PI;
use std::fmt::{Display, Formatter};

fn log_init() {
    tracing_subscriber::fmt::init();
}

fn main() -> Result<()> {
    log_init();
    disable_ffmpeg_logging();

    let yolo = YoloDetectSession::new("./data/model")?;
    let sam2 = SAMImageInferenceSession::new("./data/model/other5")?;

    let path = "./data/image/rt.jpeg";
    let image = Image::open_file(path)?;
    let sam_image = image.clone();
    let (image_width, image_height) = (image.get_width() as u32, image.get_height() as u32);

    let results = yolo.inference_yolo(image.clone(), 0.25)?;
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

    let highway = describe_anchor_points(&result_highway, image_width, image_height, "Highway");
    let sidewalk = describe_anchor_points(&result_sidewalk, image_width, image_height, "Sidewalk");

    info!("yolo highway result: {:?}", highway);
    info!("yolo sidewalk result: {:?}", sidewalk);

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

        sam2.inference_frame(
            sam_image,
            Some((1024, 1024)),
            vec![result_highway, result_sidewalk],
        )?
    };

    println!("--- Sidewalk Detections ---");
    describe_anchor_points(&result_sidewalk, image_width, image_height, "Sidewalk")
        .iter()
        .for_each(|desc| println!("{}", desc));

    println!("\n--- Highway Detections ---");
    describe_anchor_points(&result_highway, image_width, image_height, "Highway")
        .iter()
        .for_each(|desc| println!("{}", desc));

    println!("--- Analyzing Highway ---");
    // Filter highway masks: prioritize closest to user (highest average y)
    let best_highway_mask = mask[0].iter().max_by_key(|mask| {
        let center_line = extract_center_line(mask, 1024, 1024);
        center_line.first().map_or(0, |p| p.y) // Use y of nearest point
    });
    if let Some(mask) = best_highway_mask {
        println!(
            "Highway: {}",
            analyze_road_mask(mask, 1024, 1024, "Highway")
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
            analyze_road_mask(mask, 1024, 1024, "Sidewalk")
        );
    }

    Ok(())
}

// --- Configuration Constants ---
const NUM_VERTICAL_SAMPLES: u32 = 20;
const MIN_MASK_WIDTH_FOR_CENTER: u32 = 5;
const ROAD_START_Y_THRESHOLD: f32 = 0.90; // Normalized Y threshold for "at feet"
const ROAD_CENTER_X_THRESHOLD: f32 = 0.2; // Normalized X offset tolerance for "centered"
const MIN_CENTERLINE_POINTS_FOR_SHAPE: usize = 5;
const STRAIGHT_ROAD_X_DRIFT_THRESHOLD: f32 = 0.05; // Normalized X drift for straight road
const OBSTACLE_WIDTH_CHANGE_FACTOR: f32 = 0.5;
const OBSTACLE_GAP_ROWS_THRESHOLD: u32 = 3;

// --- Perspective Correction Constants ---
/// Power factor for perspective correction in distance calculation.
/// Values < 1.0 make objects near the bottom seem closer (steeper distance falloff).
/// Values > 1.0 make objects near the bottom seem relatively farther (gentler distance falloff).
/// 1.0 is linear (no perspective correction).
/// Tunable value, start around 0.6 - 0.8.
const DISTANCE_PERSPECTIVE_POWER: f32 = 0.7;

/// Threshold for height ratio (box_height / image_height) to consider an object "covering path" when very near.
const NEAR_OBJECT_HEIGHT_THRESHOLD_FACTOR: f32 = 0.3; // 30% of image height
/// Threshold for width ratio (box_width / image_width) to consider an object "covering path" when very near.
const NEAR_OBJECT_WIDTH_THRESHOLD_FACTOR: f32 = 0.5; // 50% of image width

// --- Enums and Structs ---

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
enum DirectionCategory {
    Clock(&'static str),
    Unknown,
}

impl Display for DirectionCategory {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            DirectionCategory::Clock(s) => write!(f, "{} direction", s),
            DirectionCategory::Unknown => write!(f, "Unknown direction"),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
enum DistanceCategory {
    VeryNear,       // e.g., At your feet / Underfoot
    RelativelyNear, // Close
    Near,           // Mid-range
    Far,            // Distant
    Unknown,        // Error case
}

impl Display for DistanceCategory {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            DistanceCategory::VeryNear => write!(f, "Very near"), // Base description
            DistanceCategory::RelativelyNear => write!(f, "Relatively near"),
            DistanceCategory::Near => write!(f, "Near"),
            DistanceCategory::Far => write!(f, "Far"),
            DistanceCategory::Unknown => write!(f, "Unknown distance"),
        }
    }
}

#[derive(Debug, Clone)]
struct CenterLinePoint {
    y: u32,
    center_x: f32,
    width: u32,
}

#[derive(Debug, PartialEq, Eq, Clone)]
enum RoadShape {
    Straight,
    CurvesLeft,
    CurvesRight,
    Undetermined,
}

#[derive(Debug, Clone)]
struct ObstacleInfo {
    y: u32,
    center_x: f32,
    reason: String, // Includes direction description now
}

#[derive(Debug, Clone)]
pub struct RoadAnalysisResult {
    pub starts_at_feet: bool,
    pub shape: RoadShape,
    pub obstacles: Vec<ObstacleInfo>,
    pub description: String,
}

impl Display for RoadAnalysisResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Starts at feet: {}, Shape: {:?}, Obstacles: {:?}, Description: {}",
            self.starts_at_feet, self.shape, self.obstacles, self.description
        )
    }
}

// --- Helper Functions ---

/// Calculates the perceived distance of an anchor point based on its y-coordinate.
/// Considers perspective (lower y in image means closer to the viewer).
/// Applies a non-linear mapping to account for perspective distortion.
///
/// Args:
///     y (f32): The y-coordinate of the anchor point (center of the bounding box).
///     image_height (u32): The total height of the image frame in pixels.
///
/// Returns:
///     DistanceCategory: The perceived distance category.
fn get_distance(y: f32, image_height: u32) -> DistanceCategory {
    if image_height == 0 {
        error!("Image height cannot be zero for distance calculation.");
        return DistanceCategory::Unknown;
    }

    // Normalize y coordinate to the range [0.0, 1.0]
    // 0.0 is the top of the image (far), 1.0 is the bottom (near).
    let normalized_y = (y / image_height as f32).max(0.0).min(1.0);

    // Apply perspective correction using a power function.
    // This maps the linear normalized_y to a non-linear scale.
    // With power < 1, values closer to 1.0 (bottom) increase faster,
    // simulating the visual effect of things getting much closer rapidly at the bottom.
    let perspective_adjusted_y = normalized_y.powf(DISTANCE_PERSPECTIVE_POWER);

    // Define thresholds based on the perspective-adjusted value.
    // *** These thresholds likely need tuning based on the chosen power factor ***
    // *** and the desired feel of "near" vs "far". Experimentation is key! ***
    if perspective_adjusted_y > 0.90 {
        // Adjusted from 0.85
        DistanceCategory::VeryNear
    } else if perspective_adjusted_y > 0.75 {
        // Adjusted from 0.60
        DistanceCategory::RelativelyNear
    } else if perspective_adjusted_y > 0.50 {
        // Adjusted from 0.35
        DistanceCategory::Near
    } else {
        DistanceCategory::Far
    }
}

/// Calculates the perceived direction of a point based on its angle relative
/// to an origin point (defaults to center bottom of the frame).
///
/// Args:
///     x (f32): The x-coordinate of the target point.
///     y (f32): The y-coordinate of the target point (0 is top, image_height is bottom).
///     origin (Option<(f32, f32)>): The (x, y) origin for angle calculation. If None, uses image bottom-center.
///     image_width (u32): The total width of the image frame in pixels.
///     image_height (u32): The total height of the image frame in pixels.
///
/// Returns:
///     DirectionCategory: The direction represented as a clock face position (9 to 3 o'clock).
fn get_direction(
    x: f32,
    y: f32,
    origin: Option<(f32, f32)>,
    image_width: u32,
    image_height: u32,
) -> DirectionCategory {
    if image_width == 0 || image_height == 0 {
        error!("Image dimensions cannot be zero for direction calculation.");
        return DirectionCategory::Unknown;
    }

    // Define the origin: Use provided origin or default to center bottom.
    let (origin_x, origin_y) = origin.unwrap_or((image_width as f32 / 2.0, image_height as f32));

    let dx = x - origin_x;
    // dy is calculated such that positive values mean "up" from the origin towards the top of the image.
    let dy = origin_y - y;

    // Handle edge cases where dy is very close to zero or slightly negative
    // atan2 handles dy=0 correctly, mapping to +/- PI/2 (3/9 o'clock).
    // atan2(0, 0) -> 0 (12 o'clock).
    if dy < 0.0 && dy.abs() < 1e-6 {
        // Effectively on the origin's horizontal line, treat dy as 0 for atan2.
    } else if dy < 0.0 {
        // Point is below the origin line. This can happen if origin is not bottom of image (e.g., road mask origin).
        // The angle calculation might be less intuitive ("behind"), but atan2 handles it.
        // We clamp later to the forward arc.
        // warn!( "Point y ({}) is below the origin line y ({}), direction might be less intuitive.", y, origin_y);
    }

    // Calculate the angle using atan2(dx, dy).
    let angle_rad = dx.atan2(dy); // Range: -PI to PI

    // Clamp the angle to the forward arc [-PI/2, PI/2] (9 o'clock to 3 o'clock).
    // This ensures we only describe directions in front of the origin.
    let clamped_angle_rad = angle_rad.max(-PI / 2.0).min(PI / 2.0);

    // Map the angle [-PI/2, PI/2] to an index [0, 12] for clock strings.
    let index = (((clamped_angle_rad + PI / 2.0) / PI) * 12.0).round() as usize;
    let index = index.min(12); // Ensure index is within bounds [0, 12]

    let clock_str = match index {
        0 => "9 o'clock",
        1 => "9:30",
        2 => "10 o'clock",
        3 => "10:30",
        4 => "11 o'clock",
        5 => "11:30",
        6 => "12 o'clock",
        7 => "12:30",
        8 => "1 o'clock",
        9 => "1:30",
        10 => "2 o'clock",
        11 => "2:30",
        12 => "3 o'clock",
        _ => unreachable!(),
    };

    DirectionCategory::Clock(clock_str)
}

/// Extracts the center line and width profile of the road mask.
fn extract_center_line(mask: &BitVec, image_width: u32, image_height: u32) -> Vec<CenterLinePoint> {
    let mut center_line = Vec::new();
    if image_width == 0 || image_height == 0 || mask.is_empty() {
        return center_line;
    }
    if mask.len() != (image_width * image_height) as usize {
        error!(
            "Mask length mismatch: {} vs {}",
            mask.len(),
            image_width * image_height
        );
        return center_line;
    }

    let y_step = (image_height / (NUM_VERTICAL_SAMPLES + 1)).max(1); // Ensure step is at least 1

    for i in 1..=NUM_VERTICAL_SAMPLES {
        let y = image_height.saturating_sub(i * y_step); // Sample from bottom up
        if y >= image_height {
            continue;
        } // Should not happen with saturating_sub

        let row_start_index = (y * image_width) as usize;
        let mut min_x: Option<u32> = None;
        let mut max_x: Option<u32> = None;

        for x in 0..image_width {
            let index = row_start_index + x as usize;
            // Double check bounds just in case, though logic should prevent issues
            if index < mask.len() && mask[index] {
                if min_x.is_none() {
                    min_x = Some(x);
                }
                max_x = Some(x);
            }
        }

        if let (Some(min_val), Some(max_val)) = (min_x, max_x) {
            let width = max_val.saturating_sub(min_val) + 1;
            if width >= MIN_MASK_WIDTH_FOR_CENTER {
                let center_x = (min_val + max_val) as f32 / 2.0;
                center_line.push(CenterLinePoint { y, center_x, width });
            }
        }
    }
    // Resulting points are ordered near (high y) to far (low y)
    center_line
}

/// Analyzes the shape (straight, curve left/right) based on centerline points.
fn analyze_centerline_shape(centerline: &[CenterLinePoint], image_width: u32) -> RoadShape {
    if centerline.len() < MIN_CENTERLINE_POINTS_FOR_SHAPE {
        return RoadShape::Undetermined;
    }
    let nearest_point = &centerline[0];
    let farthest_point = centerline.last().unwrap();
    let normalized_drift = (farthest_point.center_x - nearest_point.center_x) / image_width as f32;

    if normalized_drift.abs() < STRAIGHT_ROAD_X_DRIFT_THRESHOLD {
        RoadShape::Straight
    } else if normalized_drift > 0.0 {
        // Farthest point is right -> Curves Right
        RoadShape::CurvesRight
    } else {
        // Farthest point is left -> Curves Left
        RoadShape::CurvesLeft
    }
}

/// Detects obstacles like gaps or significant narrowing based on the center line profile.
/// Obstacle direction is calculated relative to the road's starting point.
fn detect_obstacles(
    center_line: &[CenterLinePoint],
    image_width: u32,
    image_height: u32,
) -> Vec<ObstacleInfo> {
    let mut obstacles = Vec::new();
    if center_line.len() < 2 {
        return obstacles;
    }

    let y_step = (image_height / (NUM_VERTICAL_SAMPLES + 1)).max(1);
    let origin_point = &center_line[0]; // Use the nearest point on the centerline as the origin for obstacle directions

    // Detect Gaps
    let mut consecutive_missing_rows = 0;
    let mut last_valid_y = origin_point.y; // Keep track of the last row where the mask was found

    for i in 1..=NUM_VERTICAL_SAMPLES {
        let current_y = image_height.saturating_sub(i * y_step);
        let sampled_point_exists = center_line
            .iter()
            .any(|p| p.y.abs_diff(current_y) < y_step / 2); // Check if a point exists near this y

        if sampled_point_exists {
            // Update last known valid y if we find the mask again
            if let Some(p) = center_line
                .iter()
                .find(|p| p.y.abs_diff(current_y) < y_step / 2)
            {
                last_valid_y = p.y;
            }
            consecutive_missing_rows = 0; // Reset gap counter
        } else {
            consecutive_missing_rows += 1;
            if consecutive_missing_rows == OBSTACLE_GAP_ROWS_THRESHOLD {
                // Estimate gap position: halfway through the missing rows, below the last seen row
                let gap_y_estimate =
                    last_valid_y.saturating_sub((OBSTACLE_GAP_ROWS_THRESHOLD * y_step) / 2);

                // Find the center_x of the point just before the gap started
                let point_before_gap = center_line.iter().find(|p| p.y == last_valid_y);
                let center_x_estimate =
                    point_before_gap.map_or(origin_point.center_x, |p| p.center_x);

                let obstacle_direction = get_direction(
                    center_x_estimate,
                    gap_y_estimate as f32,
                    Some((origin_point.center_x, origin_point.y as f32)), // Relative to road start
                    image_width,
                    image_height,
                );

                obstacles.push(ObstacleInfo {
                    y: gap_y_estimate,
                    center_x: center_x_estimate,
                    reason: format!("Potential gap near {}", obstacle_direction),
                });
                // Don't reset consecutive_missing_rows here, let it continue counting if the gap is larger
            }
        }
    }

    // Detect Narrowing
    for i in 0..(center_line.len() - 1) {
        let p_near = &center_line[i]; // Closer to viewer (higher y)
        let p_far = &center_line[i + 1]; // Further from viewer (lower y)

        // Check if width decreased significantly going further away
        if (p_far.width as f32) < (p_near.width as f32 * OBSTACLE_WIDTH_CHANGE_FACTOR) {
            // Check if this narrowing isn't already marked as part of a gap
            let y_mid_narrowing = (p_near.y + p_far.y) / 2;
            let is_near_gap = obstacles.iter().any(
                |obs| obs.reason.contains("gap") && obs.y.abs_diff(y_mid_narrowing) < y_step * 2, // Generous window around gap Y
            );

            if !is_near_gap {
                let obstacle_direction = get_direction(
                    p_far.center_x, // Use the position where narrowing is confirmed
                    p_far.y as f32,
                    Some((origin_point.center_x, origin_point.y as f32)), // Relative to road start
                    image_width,
                    image_height,
                );
                obstacles.push(ObstacleInfo {
                    y: p_far.y, // Position where the narrowing is significant
                    center_x: p_far.center_x,
                    reason: format!("Significant narrowing near {}", obstacle_direction),
                });
            }
        }
    }

    // Sort obstacles by Y coordinate (descending: nearer first)
    obstacles.sort_by(|a, b| b.y.cmp(&a.y));

    obstacles
}

/// Checks if the road mask starts near the bottom edge of the image and is horizontally centered.
/// Returns a tuple containing:
/// - bool: whether the road starts at feet
/// - Option<DirectionCategory>: the clock direction if not starting ideally at feet/center
fn check_road_start(
    center_line: &[CenterLinePoint],
    image_width: u32,
    image_height: u32,
) -> (bool, Option<DirectionCategory>) {
    match center_line.first() {
        Some(nearest_point) => {
            // Vertical check: Is the nearest point close enough to the bottom?
            let normalized_y = nearest_point.y as f32 / image_height as f32;
            let is_at_bottom = normalized_y >= ROAD_START_Y_THRESHOLD;

            // Horizontal check: Is the nearest point reasonably centered horizontally?
            let normalized_x_offset = (nearest_point.center_x / image_width as f32 - 0.5).abs();
            let is_centered = normalized_x_offset <= ROAD_CENTER_X_THRESHOLD;

            if is_at_bottom && is_centered {
                (true, None) // Ideal start: at feet and centered
            } else {
                // Not ideal start, calculate direction relative to image bottom-center
                let direction = get_direction(
                    nearest_point.center_x,
                    nearest_point.y as f32,
                    None, // Use default image bottom-center origin
                    image_width,
                    image_height,
                );
                (false, Some(direction))
            }
        }
        None => (false, None), // No centerline points found
    }
}

// --- Main Analysis Function ---

/// Analyzes a road mask to determine shape, obstacles, and starting position.
pub fn analyze_road_mask(
    mask: &BitVec,
    image_width: u32,
    image_height: u32,
    object_type_name: &str,
) -> RoadAnalysisResult {
    if image_width == 0
        || image_height == 0
        || mask.is_empty()
        || mask.len() != (image_width * image_height) as usize
    {
        return RoadAnalysisResult {
            starts_at_feet: false,
            shape: RoadShape::Undetermined,
            obstacles: vec![],
            description: format!(
                "Error: Invalid input mask or dimensions for {}.",
                object_type_name
            ),
        };
    }

    let center_line_points = extract_center_line(mask, image_width, image_height);

    if center_line_points.is_empty() {
        return RoadAnalysisResult {
            starts_at_feet: false,
            shape: RoadShape::Undetermined,
            obstacles: vec![],
            description: format!("No significant {} found in the mask.", object_type_name),
        };
    }

    let (starts_at_feet, start_direction_issue) =
        check_road_start(&center_line_points, image_width, image_height);
    let shape = analyze_centerline_shape(&center_line_points, image_width);
    let obstacles = detect_obstacles(&center_line_points, image_width, image_height);

    // Generate Description
    let mut description_parts = Vec::new();

    // Start position description
    if starts_at_feet {
        description_parts.push(format!("The {} starts at your feet.", object_type_name));
    } else {
        // Use perspective-corrected distance for the warning message
        let nearest_point = center_line_points.first().unwrap(); // Safe due to earlier check
        let start_distance = get_distance(nearest_point.y as f32, image_height);
        let direction_str = match start_direction_issue {
            Some(dir) => format!("{}", dir),
            None => "centered".to_string(), // Should have direction if not starts_at_feet, but fallback
        };
        description_parts.push(format!(
            "Warning: The {} doesn't start directly underfoot. It appears {} in the {}.",
            object_type_name,
            start_distance, // e.g., "Relatively near", "Near"
            direction_str
        ));
    }

    // Shape description
    match shape {
        RoadShape::Straight => description_parts.push("It proceeds straight.".to_string()),
        RoadShape::CurvesLeft => description_parts.push("It curves left.".to_string()),
        RoadShape::CurvesRight => description_parts.push("It curves right.".to_string()),
        RoadShape::Undetermined => description_parts.push("Its shape is unclear.".to_string()),
    }

    // Obstacle description
    if obstacles.is_empty() {
        description_parts.push("No immediate obstacles detected.".to_string());
    } else {
        // Describe the nearest obstacle first
        let first_obstacle = &obstacles[0];
        let obstacle_distance = get_distance(first_obstacle.y as f32, image_height); // Use perspective distance
        let obstacle_desc = format!(
            "Obstacle ({}) detected, {}.", // Reason now includes direction relative to road start
            first_obstacle.reason,
            obstacle_distance // Add distance category
        );
        description_parts.push(obstacle_desc);
        if obstacles.len() > 1 {
            description_parts.push(format!(
                "{} other potential obstacles further down.", // Changed wording slightly
                obstacles.len() - 1
            ));
        }
    }

    // --- User Position & Sidewalk Warnings (Example Logic) ---
    // This requires knowing the user's approximate position relative to the mask.
    // Let's assume a hypothetical user_x, user_y (e.g., bottom center of image).
    // We'd need to check if (user_x, user_y) falls within the sidewalk mask.

    let user_x = image_width as f32 / 2.0;
    let user_y = image_height as f32 - 1.0; // Assume user is at the very bottom center pixel

    if object_type_name == "Sidewalk" {
        let user_on_sidewalk = mask
            .get((user_y as u32 * image_width + user_x as u32) as usize)
            .map_or(false, |bit| *bit);

        if !user_on_sidewalk && !center_line_points.is_empty() {
            // User is off the sidewalk, but it's visible. Provide guidance.
            let nearest_sidewalk_point = center_line_points.first().unwrap();
            let direction_to_sidewalk = get_direction(
                nearest_sidewalk_point.center_x,
                nearest_sidewalk_point.y as f32,
                Some((user_x, user_y)), // Direction from user to nearest sidewalk point
                image_width,
                image_height,
            );
            description_parts.push(format!(
                "Warning: You are off the sidewalk. The sidewalk is in the {}.",
                direction_to_sidewalk
            ));
        }
        // Add logic here to check if user is near the edge of the sidewalk mask
        // (e.g., check pixels near user_x, user_y) and issue "about to leave" warning.
        // This requires analyzing mask connectivity near the user's position.
        // Example: Check if mask[user_pos] is true, but mask[user_pos - offset] or mask[user_pos + offset] is false.
    }

    let description = description_parts.join(" ");

    RoadAnalysisResult {
        starts_at_feet,
        shape,
        obstacles,
        description,
    }
}

/// Generates textual descriptions for detected objects (anchor points).
/// Incorporates perspective-corrected distance, direction, and considers bounding box size
/// for objects very close to the user ("underfoot" condition).
///
/// Args:
///     results (&[YoloDetectResult]): Slice of detection results.
///     image_width (u32): Width of the image frame.
///     image_height (u32): Height of the image frame.
///     object_type_name (&str): Name for the object type (e.g., "Highway", "Sidewalk").
///
/// Returns:
///     Vec<String>: A list of textual descriptions for each result.
pub fn describe_anchor_points(
    results: &[YoloDetectResult],
    image_width: u32,
    image_height: u32,
    object_type_name: &str,
) -> Vec<String> {
    let mut descriptions = Vec::new();

    if image_width == 0 || image_height == 0 {
        error!("Image dimensions are zero. Cannot generate anchor point descriptions.");
        return descriptions;
    }

    if results.is_empty() {
        // info!("No {} detected to describe.", object_type_name); // Optional info message
        return descriptions;
    }

    for (i, result) in results.iter().enumerate() {
        // Calculate perspective-corrected distance and direction relative to image bottom-center
        let distance = get_distance(result.y, image_height);
        let direction = get_direction(result.x, result.y, None, image_width, image_height);

        let description: String;

        // --- Perspective Adjustment for "Very Near" Objects ---
        // Check if the object is very close AND large enough to warrant "underfoot" description
        if distance == DistanceCategory::VeryNear {
            let box_height_ratio = result.height / image_height as f32;
            let box_width_ratio = result.width / image_width as f32;

            // If the box is tall (covers significant vertical space near user)
            // AND wide (covers significant horizontal space near user)
            if box_height_ratio >= NEAR_OBJECT_HEIGHT_THRESHOLD_FACTOR
                && box_width_ratio >= NEAR_OBJECT_WIDTH_THRESHOLD_FACTOR
            {
                // Special description for large, close objects (like the road itself)
                description = format!(
                    "{} {}: Covering the path underfoot, primarily in the {}", // Changed wording slightly
                    object_type_name,
                    i + 1,
                    direction // Direction is still useful (e.g., underfoot and slightly left)
                );
            } else {
                // Standard description for objects that are "Very near" but not huge
                description = format!(
                    "{} {}: {}, {}", // Uses standard "Very near" from Display trait
                    object_type_name,
                    i + 1,
                    distance,
                    direction
                );
            }
        } else {
            // Standard description for objects not classified as "Very near"
            description = format!(
                "{} {}: {}, {}",
                object_type_name,
                i + 1,
                distance, // Near, Far, etc.
                direction
            );
        }

        descriptions.push(description);
    }

    descriptions
}
