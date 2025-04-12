use crate::detect::analysis::Describer;
use crate::detect::constants::{
    EDGE_PROXIMITY_THRESHOLD_FACTOR, MIN_SIDEWALK_WIDTH_FACTOR_FOR_EDGE_WARNING,
};
use crate::detect::property::analyse_result::RoadAnalysisData;
use crate::detect::property::direction::DirectionCategory;
use crate::detect::property::distance::DistanceCategory;
use log::debug;
use std::fmt::Write;

pub struct UserPositionDescriber;

impl Describer for UserPositionDescriber {
    fn describe(&self, data: &RoadAnalysisData, object_type_name: &str) -> Option<String> {
        // This logic is primarily for sidewalks and requires a centerline.
        // Only run if the object is explicitly a "Sidewalk".
        if object_type_name != "Sidewalk" || data.center_lines.is_empty() {
            return None;
        }

        let mut messages = Vec::new();

        let user_x = data.image_width as f32 / 2.0;
        let user_y = data.image_height as f32 - 1.0; // Assume user is at the very bottom center

        // Get the nearest point of the sidewalk centerline
        // We checked !is_empty() above, so unwrap is safe.
        let nearest_sidewalk_point = data.center_lines.first().unwrap();

        // Estimate if user is vertically close enough to be considered "on" the start
        // We use the pre-calculated `starts_at_feet` which checks vertical proximity.
        let vertically_close = data.starts_at_feet;

        // Check horizontal position relative to the nearest centerline point
        let half_width = nearest_sidewalk_point.width as f32 / 2.0;
        let sidewalk_left_edge = nearest_sidewalk_point.center_x - half_width;
        let sidewalk_right_edge = nearest_sidewalk_point.center_x + half_width;

        let user_is_horizontally_on_sidewalk =
            user_x >= sidewalk_left_edge && user_x <= sidewalk_right_edge;

        debug!(
            "User Position Check: vertically_close={}, user_x={}, sidewalk_center={}, sidewalk_width={}, left_edge={}, right_edge={}, user_horizontally_on={}",
            vertically_close, user_x, nearest_sidewalk_point.center_x, nearest_sidewalk_point.width, sidewalk_left_edge, sidewalk_right_edge, user_is_horizontally_on_sidewalk
        );

        if vertically_close && user_is_horizontally_on_sidewalk {
            // --- User is ON the sidewalk ---
            // Now check proximity to the edge

            // Avoid edge warnings on very narrow sidewalks
            let min_width_for_warning =
                data.image_width as f32 * MIN_SIDEWALK_WIDTH_FACTOR_FOR_EDGE_WARNING;
            if nearest_sidewalk_point.width as f32 >= min_width_for_warning && half_width > 1.0 {
                // Check half_width > 0 avoids division by zero if width is tiny
                let distance_from_center = (user_x - nearest_sidewalk_point.center_x).abs();
                let edge_proximity_threshold = half_width * EDGE_PROXIMITY_THRESHOLD_FACTOR;

                if distance_from_center >= edge_proximity_threshold {
                    // User is near an edge
                    if user_x < nearest_sidewalk_point.center_x {
                        messages.push(
                            "Warning: You are near the left edge of the sidewalk".to_string(),
                        );
                    } else {
                        messages.push(
                            "Warning: You are near the right edge of the sidewalk".to_string(),
                        );
                    }
                } else {
                    // User is relatively centered, no message needed unless explicitly desired.
                    // messages.push("You are centered on the sidewalk.".to_string()); // Optional: Add if needed
                    debug!("User is centered on the sidewalk.");
                }
            } else {
                // Sidewalk is too narrow for edge warnings, assume user is just "on" it.
                debug!("Sidewalk too narrow for edge warning.");
            }
        } else {
            // --- User is OFF the sidewalk (or sidewalk doesn't start at feet) ---
            // Check if the sidewalk is visible at all (i.e., centerline exists)
            // Since we passed the `is_empty()` check earlier, we know it's visible.

            // Provide guidance towards the sidewalk.
            let direction_to_sidewalk = DirectionCategory::get_direction(
                nearest_sidewalk_point.center_x, // Target X
                nearest_sidewalk_point.y as f32, // Target Y
                Some((user_x, user_y)),          // Origin: User's position
                data.image_width,
                data.image_height,
            );
            let distance_to_sidewalk = DistanceCategory::get_distance(
                nearest_sidewalk_point.y as f32, // Use sidewalk's Y for distance perception
                data.image_height,
            );

            // Refine message based on whether it starts at feet but user is off horizontally,
            // or if it doesn't start at feet at all.
            if !vertically_close {
                // The RoadStartDescriber already warns if it doesn't start underfoot.
                // Avoid duplicating that specific warning. Maybe add context?
                // Or let RoadStartDescriber handle the "not starting at feet" part.
                // Let's assume RoadStartDescriber handles the vertical distance warning.
                // We can still add guidance if the user is *also* horizontally off.
                if !user_is_horizontally_on_sidewalk {
                    messages.push(format!(
                        "The sidewalk starts {} ahead and is in the {}",
                        distance_to_sidewalk,  // e.g., "Near"
                        direction_to_sidewalk  // e.g., "11 o'clock direction"
                    ));
                } else {
                    // Starts ahead, but user *would* be horizontally aligned if they moved forward.
                    // RoadStartDescriber likely covers this.
                    debug!("Sidewalk starts ahead, but user is horizontally aligned.");
                }
            } else {
                // Starts at feet vertically, but user is horizontally off.
                messages.push(format!(
                    "Warning: You appear to be off the sidewalk. The sidewalk is in the {}",
                    direction_to_sidewalk
                ));
            }
        }

        if messages.is_empty() {
            None
        } else {
            // Combine messages if multiple warnings were generated (unlikely with current logic, but good practice)
            let mut description = String::new();
            write!(description, "{}", messages.join(". ")).unwrap();
            Some(description)
        }
    }
}
