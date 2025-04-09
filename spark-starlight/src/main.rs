#![feature(let_chains)]
#![cfg_attr(debug_assertions, allow(warnings))]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Result;
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
use spark_inference::utils::masks::ApplyMask;
use spark_media::filter::filter::AVFilter;
use spark_media::{Image, RGB};
use std::fmt::{Display, Formatter};

fn log_init() {
    tracing_subscriber::fmt::init();
}

fn main() -> Result<()> {
    log_init();
    disable_ffmpeg_logging();

    let yolo = YoloDetectSession::new("./data/model")?;
    let sam2 = SAMImageInferenceSession::new("./data/model/other5")?;

    let path = "./data/image/d4.jpg";
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

    info!("yolo highway result: {:?}", result_highway);
    info!("yolo sidewalk result: {:?}", result_sidewalk);

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
    mask[0].iter().for_each(|x| {
        let result_straight = analyze_road_mask(x, 1024, 1024, "Highway");
        println!("Starts at feet: {}", result_straight.starts_at_feet);
        println!("Shape: {:?}", result_straight.shape);
        println!("Obstacles: {:?}", result_straight.obstacles);
        println!("Description: {}", result_straight.description);
    });

    println!("\n--- Analyzing Sidewalk ---");
    mask[1].iter().for_each(|x| {
        let result_curve_gap = analyze_road_mask(x, 1024, 1024, "Sidewalk");
        println!("Starts at feet: {}", result_curve_gap.starts_at_feet);
        println!("Shape: {:?}", result_curve_gap.shape);
        println!("Obstacles: {:?}", result_curve_gap.obstacles);
        println!("Description: {}", result_curve_gap.description);
    });

    Ok(())
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
enum DirectionCategory {
    Clock(String),
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

// Enum for distance categories for better code readability
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
enum DistanceCategory {
    VeryNear, // e.g., At your feet
    RelativelyNear,
    Near, // Mid-range
    Far,
    Unknown, // Error case
}

// Implement Display trait for easy conversion to String
impl Display for DistanceCategory {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            DistanceCategory::VeryNear => write!(f, "Very near"),
            DistanceCategory::RelativelyNear => write!(f, "Relatively near"),
            DistanceCategory::Near => write!(f, "Near"),
            DistanceCategory::Far => write!(f, "Far"),
            DistanceCategory::Unknown => write!(f, "Unknown distance"),
        }
    }
}

/// Calculates the perceived distance of an anchor point based on its y-coordinate.
/// Considers perspective (lower y in image means closer to the viewer).
///
/// Args:
///     y (f32): The y-coordinate of the anchor point (center of the bounding box).
///     image_height (u32): The total height of the image frame in pixels.
///
/// Returns:
///     DistanceCategory: The perceived distance category.
fn get_distance(y: f32, image_height: u32) -> DistanceCategory {
    if image_height == 0 {
        eprintln!("Error: Image height cannot be zero.");
        return DistanceCategory::Unknown;
    }

    // Normalize y coordinate to the range [0.0, 1.0]
    // 0.0 is the top of the image (far), 1.0 is the bottom (near).
    // Clamp to ensure it stays within bounds.
    let normalized_y = (y / image_height as f32).max(0.0).min(1.0);

    // Define thresholds for distance categories. These may need tuning.
    // Higher normalized_y means closer to the bottom edge of the frame (nearer).
    if normalized_y > 0.85 {
        // Bottom 15% of the image height
        DistanceCategory::VeryNear // "At your feet" can also be used
    } else if normalized_y > 0.60 {
        // Roughly between 60% and 85% down the frame
        DistanceCategory::RelativelyNear
    } else if normalized_y > 0.35 {
        // Roughly between 35% and 60% down the frame
        DistanceCategory::Near
    } else {
        // Top 35% of the image height
        DistanceCategory::Far
    }
}

// Assume get_direction function is available (copied here for completeness)
fn get_direction(x: f32, image_width: u32) -> DirectionCategory {
    if image_width == 0 {
        return DirectionCategory::Unknown;
    }
    let normalized_x = (x / image_width as f32).max(0.0).min(1.0);
    let index = (normalized_x * 12.0).round() as usize;
    let index = index.min(12);
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
        _ => return DirectionCategory::Unknown,
    };
    DirectionCategory::Clock(clock_str.to_string())
}

// --- New Structs and Enums for Mask Analysis ---

#[derive(Debug, Clone)]
struct CenterlinePoint {
    y: u32,        // Vertical position (pixel row)
    center_x: f32, // Horizontal center of the mask at this row
    width: u32,    // Width of the mask at this row
}

#[derive(Debug, PartialEq, Eq, Clone)]
enum RoadShape {
    Straight,
    CurvesLeft, // Extends towards the left in the distance (drifts right as it gets closer)
    CurvesRight, // Extends towards the right in the distance (drifts left as it gets closer)
    Undetermined, // Not enough data or too complex
}

#[derive(Debug, Clone)]
struct ObstacleInfo {
    y: u32,         // Approximate vertical position of the obstacle
    center_x: f32,  // Approximate horizontal position
    reason: String, // e.g., "Gap", "Narrowing"
}

#[derive(Debug, Clone)]
pub struct RoadAnalysisResult {
    pub starts_at_feet: bool,
    pub shape: RoadShape,
    pub obstacles: Vec<ObstacleInfo>,
    pub description: String, // Generated textual summary
}

// --- Configuration Constants (Crucial for Tuning!) ---
const NUM_VERTICAL_SAMPLES: u32 = 20; // How many horizontal slices to analyze
const MIN_MASK_WIDTH_FOR_CENTER: u32 = 5; // Minimum pixels wide to calculate a center
const ROAD_START_Y_THRESHOLD: f32 = 0.90; // How close to the bottom (normalized) road should start
const MIN_CENTERLINE_POINTS_FOR_SHAPE: usize = 5; // Need at least this many points for shape analysis
const STRAIGHT_ROAD_X_DRIFT_THRESHOLD: f32 = 0.05; // Max normalized horizontal drift for "straight"
const OBSTACLE_WIDTH_CHANGE_FACTOR: f32 = 0.5; // Significant narrowing if width drops by this factor
const OBSTACLE_GAP_ROWS_THRESHOLD: u32 = 3; // How many consecutive empty sampled rows indicate a gap

// --- Helper Functions ---

/// Extracts the centerline and width profile of the road mask.
/// Assumes mask is row-major: index = y * width + x
fn extract_centerline(mask: &BitVec, image_width: u32, image_height: u32) -> Vec<CenterlinePoint> {
    let mut centerline = Vec::new();
    if image_width == 0 || image_height == 0 || mask.is_empty() {
        return centerline; // Cannot process if dimensions are zero or mask is empty
    }
    if mask.len() != (image_width * image_height) as usize {
        eprintln!(
            "Warning: Mask length ({}) does not match image dimensions ({}x{}={})",
            mask.len(),
            image_width,
            image_height,
            image_width * image_height
        );
        // Decide how to handle: return empty, panic, or try to proceed? Returning empty is safer.
        return centerline;
    }

    let y_step = image_height / (NUM_VERTICAL_SAMPLES + 1); // +1 to avoid sampling very top/bottom edges potentially

    for i in 1..=NUM_VERTICAL_SAMPLES {
        let y = image_height - i * y_step; // Sample from bottom up (near to far)
        if y >= image_height {
            continue;
        } // Avoid potential index out of bounds on the last step

        let mut min_x: Option<u32> = None;
        let mut max_x: Option<u32> = None;

        let row_start_index = (y * image_width) as usize;

        // Find the leftmost and rightmost 'true' bits in the current row
        for x in 0..image_width {
            let index = row_start_index + x as usize;
            // Check bounds before accessing mask - Belt and suspenders approach
            if index < mask.len() && mask[index] {
                if min_x.is_none() {
                    min_x = Some(x);
                }
                max_x = Some(x);
            }
        }

        // If we found a valid segment in this row, calculate center and width
        if let (Some(min_val), Some(max_val)) = (min_x, max_x) {
            let width = max_val.saturating_sub(min_val) + 1;
            if width >= MIN_MASK_WIDTH_FOR_CENTER {
                let center_x = (min_val + max_val) as f32 / 2.0;
                centerline.push(CenterlinePoint { y, center_x, width });
            }
            // Optional: else { maybe log that a thin segment was ignored }
        }
        // Optional: else { maybe log that row 'y' had no mask pixels }
    }

    // Resulting centerline points are ordered from near (high y) to far (low y)
    centerline
}

/// Analyzes the shape (straight, curve left/right) based on centerline points.
fn analyze_centerline_shape(centerline: &[CenterlinePoint], image_width: u32) -> RoadShape {
    if centerline.len() < MIN_CENTERLINE_POINTS_FOR_SHAPE {
        return RoadShape::Undetermined;
    }

    // Compare the horizontal position of the nearest point to the farthest point
    // Points are ordered near to far (high y to low y)
    let nearest_point = &centerline[0];
    let farthest_point = centerline.last().unwrap(); // Safe due to length check above

    // Calculate horizontal drift, normalized by image width
    // Positive drift: Farthest point is to the right of the nearest point (relative to image coords)
    // Negative drift: Farthest point is to the left of the nearest point
    let normalized_drift = (farthest_point.center_x - nearest_point.center_x) / image_width as f32;

    if normalized_drift.abs() < STRAIGHT_ROAD_X_DRIFT_THRESHOLD {
        RoadShape::Straight
    } else if normalized_drift > 0.0 {
        // Farthest point is right -> Road extends to the right -> Curves Right from user perspective
        RoadShape::CurvesRight
    } else {
        // Farthest point is left -> Road extends to the left -> Curves Left from user perspective
        RoadShape::CurvesLeft
    }
}

/// Detects obstacles like gaps or significant narrowing based on the centerline profile.
fn detect_obstacles(
    centerline: &[CenterlinePoint],
    image_width: u32,
    image_height: u32, // Needed for gap detection logic potentially
    mask: &BitVec,     // Needed for more detailed gap check if required
) -> Vec<ObstacleInfo> {
    let mut obstacles = Vec::new();
    if centerline.len() < 2 {
        // Need at least two points to see changes
        return obstacles;
    }

    let mut consecutive_missing_rows = 0;
    let mut last_sampled_y = image_height; // Start from bottom

    // Check for gaps *between* sampled rows
    let y_step = image_height / (NUM_VERTICAL_SAMPLES + 1);
    for i in 1..=NUM_VERTICAL_SAMPLES {
        let current_y = image_height - i * y_step;
        let sampled_point_exists = centerline.iter().any(|p| p.y == current_y);

        if sampled_point_exists {
            consecutive_missing_rows = 0; // Reset gap counter
        } else {
            consecutive_missing_rows += 1;
            if consecutive_missing_rows == OBSTACLE_GAP_ROWS_THRESHOLD {
                // Found a potential gap. Report it roughly in the middle of the gap.
                let gap_y = current_y + (OBSTACLE_GAP_ROWS_THRESHOLD / 2) * y_step;
                // Try to find the center_x from the last valid point before the gap
                let last_valid_point = centerline.iter().find(|p| p.y > gap_y); // Find point just below the gap
                let center_x = last_valid_point.map_or(image_width as f32 / 2.0, |p| p.center_x); // Default to center if no prior point

                obstacles.push(ObstacleInfo {
                    y: gap_y,
                    center_x,
                    reason: "Potential gap".to_string(),
                });
            }
        }
        last_sampled_y = current_y;
    }

    // Check for significant narrowing by comparing adjacent points in the centerline
    // Iterate pairwise over centerline points (near to far)
    for i in 0..(centerline.len() - 1) {
        let p_near = &centerline[i]; // Point closer to user
        let p_far = &centerline[i + 1]; // Point farther away

        // Expect width to decrease as we go farther (p_far.width <= p_near.width generally)
        // An obstacle is suggested if the far point is *significantly* narrower than expected
        // Simple check: if p_far is much narrower than p_near
        if (p_far.width as f32) < (p_near.width as f32 * OBSTACLE_WIDTH_CHANGE_FACTOR) {
            // Check if this obstacle is already captured by a gap report near the same location
            let y_mid = (p_near.y + p_far.y) / 2;
            if !obstacles
                .iter()
                .any(|obs| obs.reason.contains("gap") && obs.y.abs_diff(y_mid) < y_step)
            {
                obstacles.push(ObstacleInfo {
                    y: p_far.y, // Report obstacle at the location of the narrowing
                    center_x: p_far.center_x,
                    reason: "Significant narrowing".to_string(),
                });
            }
        }
        // Optional: Add check for sudden *widening* if needed
    }

    obstacles
}

/// Checks if the road mask starts near the bottom edge of the image.
fn check_road_start(
    centerline: &[CenterlinePoint], // Assumes ordered near to far
    image_height: u32,
) -> bool {
    match centerline.first() {
        Some(nearest_point) => {
            let normalized_y = nearest_point.y as f32 / image_height as f32;
            normalized_y >= ROAD_START_Y_THRESHOLD
        }
        None => false, // No centerline points found, so doesn't start at feet
    }
}

// --- Main Analysis Function ---

/// Analyzes a road mask to determine shape, obstacles, and starting position.
///
/// Args:
///     mask (BitVec): The road mask (row-major).
///     image_width (u32): Width of the image.
///     image_height (u32): Height of the image.
///     object_type_name (str): Name for the object type (e.g., "Highway", "Sidewalk").
///
/// Returns:
///     RoadAnalysisResult: Contains analysis details and a textual description.
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

    // 1. Extract Centerline
    let centerline_points = extract_centerline(mask, image_width, image_height);

    if centerline_points.is_empty() {
        return RoadAnalysisResult {
            starts_at_feet: false,
            shape: RoadShape::Undetermined,
            obstacles: vec![],
            description: format!("No significant {} found in the mask.", object_type_name),
        };
    }

    // 2. Check Starting Position
    let starts_at_feet = check_road_start(&centerline_points, image_height);

    // 3. Analyze Shape
    let shape = analyze_centerline_shape(&centerline_points, image_width);

    // 4. Detect Obstacles
    let obstacles = detect_obstacles(&centerline_points, image_width, image_height, mask);

    // 5. Generate Description
    let mut description_parts = Vec::new();

    // Start position warning
    if !starts_at_feet {
        // Find the y-coordinate of the nearest point to describe how far it is
        let nearest_y_norm = centerline_points
            .first()
            .map_or(0.0, |p| p.y as f32 / image_height as f32);
        let distance_desc = if nearest_y_norm > 0.6 {
            "relatively near"
        } else if nearest_y_norm > 0.35 {
            "near"
        } else {
            "far"
        };
        description_parts.push(format!(
            "Warning: {} starts {} ahead, not directly at your feet.",
            object_type_name, distance_desc
        ));
    } else {
        description_parts.push(format!("The {} starts at your feet.", object_type_name));
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
        // Describe the first detected obstacle's location
        let first_obstacle = &obstacles[0];
        let obstacle_direction = get_direction(first_obstacle.center_x, image_width);
        let obstacle_desc = format!(
            "There is an obstacle ({}) near the {}.",
            first_obstacle.reason,
            obstacle_direction // Use the Display trait of DirectionCategory
        );
        description_parts.push(obstacle_desc);
        if obstacles.len() > 1 {
            description_parts.push(format!(
                "{} additional potential obstacles found.",
                obstacles.len() - 1
            ));
        }
    }

    let final_description = description_parts.join(" ");

    RoadAnalysisResult {
        starts_at_feet,
        shape,
        obstacles,
        description: final_description,
    }
}

pub fn describe_anchor_points(
    results: &[YoloDetectResult],
    image_width: u32,
    image_height: u32,
    object_type_name: &str, // Added parameter for context
) -> Vec<String> {
    let mut descriptions = Vec::new();

    if image_width == 0 || image_height == 0 {
        eprintln!("Warning: Image dimensions are zero. Cannot generate descriptions.");
        // Optionally return a single error message in the Vec
        // descriptions.push("Error: Invalid image dimensions.".to_string());
        return descriptions; // Return empty list
    }

    if results.is_empty() {
        // Optional: Return a message indicating nothing was detected of this type
        // descriptions.push(format!("No {} detected.", object_type_name));
        return descriptions; // Return empty list
    }

    for (i, result) in results.iter().enumerate() {
        // --- Crucial Assumption ---
        // We assume result.x and result.y are the *center* coordinates.
        // If they are top-left, you would calculate center points first:
        // let center_x = result.x + result.width / 2.0;
        // let center_y = result.y + result.height / 2.0;
        // Then pass center_x and center_y to get_direction/get_distance.
        // For this implementation, we proceed assuming x,y are already centers.

        let direction = get_direction(result.x, image_width);
        let distance = get_distance(result.y, image_height);

        // Format the final description string
        // Example: "Sidewalk 1: Very near, 12 o'clock direction"
        // Example: "Obstacle 2: Far, 9:30 direction"
        let description = format!(
            "{} {}: {}, {}",
            object_type_name, // e.g., "Highway", "Sidewalk"
            i + 1,            // Index the object if multiple are detected
            distance,         // Uses the Display trait of DistanceCategory
            direction         // Uses the Display trait of DirectionCategory
        );
        descriptions.push(description);
    }

    descriptions
}
