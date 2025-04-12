use crate::detect::analysis::Describer;
use crate::detect::property::analyse_result::RoadAnalysisData;
use crate::detect::property::direction::DirectionCategory;
use std::fmt::Write;

pub struct UserPositionDescriber;

impl Describer for UserPositionDescriber {
    fn describe(&self, data: &RoadAnalysisData, object_type_name: &str) -> Option<String> {
        // This logic is primarily for sidewalks and requires a centerline.
        if object_type_name != "Sidewalk" || data.center_lines.is_empty() {
            return None;
        }

        let mut description = String::new();
        let mut messages = Vec::new();

        let user_x = data.image_width as f32 / 2.0;
        let user_y = data.image_height as f32 - 1.0; // Assume user is at the very bottom center

        // Estimate if user is on sidewalk based on centerline start
        let nearest_sidewalk_point = data.center_lines.first().unwrap(); // Safe due to !is_empty check

        // Check if the nearest point of the centerline starts reasonably close horizontally
        let horizontal_diff = (nearest_sidewalk_point.center_x - user_x).abs();
        // Check if the nearest point's width covers the user's assumed position
        // This estimation is rough and doesn't use the full mask.
        let estimated_user_on_sidewalk = data.starts_at_feet && // Must start near feet vertically
            horizontal_diff < nearest_sidewalk_point.width as f32 / 2.0;

        if !estimated_user_on_sidewalk {
            // User is likely off the sidewalk, but it's visible. Provide guidance.
            let direction_to_sidewalk = DirectionCategory::get_direction(
                nearest_sidewalk_point.center_x,
                nearest_sidewalk_point.y as f32,
                Some((user_x, user_y)), // Direction from user to nearest sidewalk point
                data.image_width,
                data.image_height,
            );
            messages.push(format!(
                "Warning: You appear to be off the sidewalk. The sidewalk is in the {}",
                direction_to_sidewalk
            ));
        } else {
            // User is on the sidewalk, check proximity to the edge.
            // Add logic here for "about to leave" warning - this is harder without the mask.
            // We could check if the user's assumed position is near the *edge* of the nearest_sidewalk_point's width.
            let edge_proximity_threshold = nearest_sidewalk_point.width as f32 * 0.15; // e.g., within 15% of the edge width - needs tuning
            let distance_to_center = horizontal_diff;
            let half_width = nearest_sidewalk_point.width as f32 / 2.0;

            if half_width > edge_proximity_threshold && // Avoid warning on very narrow sidewalks
                distance_to_center > (half_width - edge_proximity_threshold)
            {
                messages.push("Warning: You are near the edge of the sidewalk".to_string());
            }
        }

        if messages.is_empty() {
            None
        } else {
            write!(description, "{}", messages.join(". ")).unwrap();
            Some(description)
        }
    }
}
