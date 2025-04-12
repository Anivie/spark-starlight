use crate::detect::analysis::Describer;
use crate::detect::constants::{
    CLOSE_GAP_Y_THRESHOLD_FACTOR, FAR_POINT_Y_THRESHOLD_FACTOR, MIN_POINTS_FOR_CONTINUATION,
};
use crate::detect::property::analyse_result::RoadAnalysisData;
use crate::detect::property::distance::DistanceCategory;
use std::fmt::Write;

pub struct PathEndingDescriber;

impl Describer for PathEndingDescriber {
    fn describe(&self, data: &RoadAnalysisData, object_type_name: &str) -> Option<String> {
        // This warning is primarily for sidewalks that start at the user's feet.
        if object_type_name != "Sidewalk" || !data.starts_at_feet || data.center_lines.is_empty() {
            return None;
        }

        let farthest_point = data.center_lines.last().unwrap(); // Safe due to !is_empty check

        let end_threshold_y = data.image_height as f32 * FAR_POINT_Y_THRESHOLD_FACTOR;
        let close_gap_threshold_y = data.image_height as f32 * CLOSE_GAP_Y_THRESHOLD_FACTOR;

        let mut sidewalk_ends_soon = false;

        // Check Condition 1: Farthest detected point doesn't reach far up the image
        // Remember: y=0 is top, y=image_height is bottom. Higher Y means closer.
        // So if the *farthest* point (lowest Y) is still *greater* than the threshold (meaning closer to user than threshold), it ends soon.
        if (farthest_point.y as f32) > end_threshold_y {
            // log::debug!("Path ending check 1: Farthest point y ({}) > threshold y ({})", farthest_point.y, end_threshold_y);
            sidewalk_ends_soon = true;
        }

        // Check Condition 2: Centerline is very short
        if !sidewalk_ends_soon && data.center_lines.len() < MIN_POINTS_FOR_CONTINUATION {
            // log::debug!("Path ending check 2: Centerline length ({}) < min points ({})", data.center_lines.len(), MIN_POINTS_FOR_CONTINUATION);
            sidewalk_ends_soon = true;
        }

        // Check Condition 3: A gap obstacle is detected relatively close
        if !sidewalk_ends_soon {
            if let Some(first_obstacle) = data.obstacles.first() {
                // Check if the obstacle is a gap AND its y-coordinate is "close" (greater than the close gap threshold)
                if first_obstacle.reason.contains("gap")
                    && (first_obstacle.y as f32) > close_gap_threshold_y
                {
                    // log::debug!("Path ending check 3: Close gap obstacle detected at y ({}) > threshold y ({})", first_obstacle.y, close_gap_threshold_y);
                    sidewalk_ends_soon = true;
                }
            }
        }

        if sidewalk_ends_soon {
            // Avoid redundant warnings: Check if the *first* reported obstacle is already very close.
            // The ObstacleDescriber would have already given a strong warning.
            let already_warned_by_close_obstacle = data.obstacles.first().map_or(false, |obs| {
                let obs_dist = DistanceCategory::get_distance(obs.y as f32, data.image_height);
                obs_dist == DistanceCategory::VeryNear
                    || obs_dist == DistanceCategory::RelativelyNear
            });

            if !already_warned_by_close_obstacle {
                let mut description = String::new();
                write!(
                    description,
                    "Warning: The sidewalk path ahead appears short or obstructed soon"
                )
                .unwrap();
                return Some(description);
            }
        }

        None // No "ending soon" warning needed
    }
}
