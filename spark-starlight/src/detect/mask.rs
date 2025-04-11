use crate::detect::property::analyse_result::RoadAnalysisResult;
use crate::detect::property::center_line::CenterLines;
use crate::detect::property::direction::DirectionCategory;
use crate::detect::property::distance::DistanceCategory;
use crate::detect::property::road_shape::RoadShape;
use crate::detect::{
    CLOSE_GAP_Y_THRESHOLD_FACTOR, FAR_POINT_Y_THRESHOLD_FACTOR, MIN_POINTS_FOR_CONTINUATION,
};
use bitvec::prelude::BitVec;
use log::debug;

/// Analyzes a road mask to determine shape, obstacles, and starting position.
/// Args:
///    mask (BitVec): The road mask to analyze.
///    image_width (u32): The width of the image frame.
///    image_height (u32): The height of the image frame.
///    object_type_name (str): The name of the object type (e.g., "Highway", "Sidewalk").
///
/// Returns:
///   RoadAnalysisResult: A struct containing analysis results including start position, shape, obstacles, and description.
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

    let center_line_points = CenterLines::extract_center_line(mask, image_width, image_height);

    if center_line_points.is_empty() {
        return RoadAnalysisResult {
            starts_at_feet: false,
            shape: RoadShape::Undetermined,
            obstacles: vec![],
            description: format!("No significant {} found in the mask.", object_type_name),
        };
    }

    let (starts_at_feet, start_direction_issue) =
        center_line_points.check_road_start(image_width, image_height);
    let shape = center_line_points.analyze_center_line_shape(image_width);
    let obstacles = center_line_points.detect_obstacles(image_width, image_height);

    // Generate Description
    let mut description_parts = Vec::new();

    // Start position description
    if starts_at_feet {
        description_parts.push(format!("The {} starts at your feet.", object_type_name));
    } else {
        // Use perspective-corrected distance for the warning message
        let nearest_point = center_line_points.first().unwrap(); // Safe due to earlier check
        let start_distance = DistanceCategory::get_distance(nearest_point.y as f32, image_height);
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
        let obstacle_distance =
            DistanceCategory::get_distance(first_obstacle.y as f32, image_height); // Use perspective distance
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
            let direction_to_sidewalk = DirectionCategory::get_direction(
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

    // --- Add this new block within the analyze_road_mask function ---

    // --- Sidewalk Ending Warning ---
    // This specific warning triggers ONLY if the sidewalk starts at the user's feet,
    // but the analysis suggests it ends or is obstructed relatively soon ahead.
    if object_type_name == "Sidewalk" && starts_at_feet && !center_line_points.is_empty() {
        let farthest_point = center_line_points.last().unwrap(); // Safe due to !is_empty check

        let end_threshold_y = image_height as f32 * FAR_POINT_Y_THRESHOLD_FACTOR;
        let close_gap_threshold_y = image_height as f32 * CLOSE_GAP_Y_THRESHOLD_FACTOR;

        let mut sidewalk_ends_soon = false;
        let mut ending_reason = ""; // For debugging/logging if needed

        // Check Condition 1: Farthest detected point doesn't reach far up the image
        if (farthest_point.y as f32) > end_threshold_y {
            sidewalk_ends_soon = true;
            ending_reason = "farthest point too close";
        }

        // Check Condition 2: Centerline is very short
        if !sidewalk_ends_soon && center_line_points.len() < MIN_POINTS_FOR_CONTINUATION {
            sidewalk_ends_soon = true;
            ending_reason = "centerline too short";
        }

        // Check Condition 3: A gap obstacle is detected relatively close
        if !sidewalk_ends_soon {
            if let Some(first_obstacle) = obstacles.first() {
                if first_obstacle.reason.contains("gap")
                    && (first_obstacle.y as f32) > close_gap_threshold_y
                {
                    sidewalk_ends_soon = true;
                    ending_reason = "close gap detected";
                }
            }
        }

        // If any condition met, add the warning, but avoid redundancy with existing close obstacle warnings.
        if sidewalk_ends_soon {
            debug!("Sidewalk ending condition met: {}", ending_reason); // Optional: Use logging

            // Check if a very near or relatively near obstacle is already the *first* reported obstacle.
            let already_warned_by_close_obstacle = obstacles.first().map_or(false, |obs| {
                let obs_dist = DistanceCategory::get_distance(obs.y as f32, image_height);
                // Only consider it "already warned" if the *first* obstacle is very close.
                obs_dist == DistanceCategory::VeryNear
                    || obs_dist == DistanceCategory::RelativelyNear
            });

            if !already_warned_by_close_obstacle {
                description_parts.push(
                    "Warning: The sidewalk path ahead appears short or obstructed soon."
                        .to_string(),
                );
            } else {
                debug!(
                    "Sidewalk ending warning suppressed due to existing close obstacle warning."
                ); // Optional: Use logging
            }
        }
    }

    let description = description_parts.join(" ");

    RoadAnalysisResult {
        starts_at_feet,
        shape,
        obstacles,
        description,
    }
}

pub fn get_best_highway(masks: &Vec<BitVec>) -> Option<&BitVec> {
    masks.iter().max_by_key(|mask| {
        let center_line = CenterLines::extract_center_line(mask, 1024, 1024);
        center_line.first().map_or(0, |p| p.y) // Use y of nearest point
    })
}
