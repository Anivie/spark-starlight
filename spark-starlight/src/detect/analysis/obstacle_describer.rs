use crate::detect::analysis::Describer;
use crate::detect::property::analyse_result::RoadAnalysisData;
use crate::detect::property::distance::DistanceCategory;
use std::fmt::Write;

pub struct ObstacleDescriber;

impl Describer for ObstacleDescriber {
    fn describe(&self, data: &RoadAnalysisData, _object_type_name: &str) -> Option<String> {
        // Only describe centerline-derived obstacles if a centerline exists.
        if data.center_lines.is_empty() || data.obstacles.is_empty() {
            // If no obstacles, we can potentially add a "clear path" message,
            // but let's keep it focused on describing *found* obstacles for now.
            // The CompositeDescriber could add a default "no obstacles" message if needed.
            // Let's return a positive confirmation if centerline exists but no obstacles were found on it.
            if !data.center_lines.is_empty() {
                return Some("No immediate obstacles detected on the path".to_string());
            } else {
                return None; // No centerline, so no centerline obstacles to describe.
            }
        }

        let mut description = String::new();
        // Describe the nearest obstacle first
        let first_obstacle = &data.obstacles[0];
        let obstacle_distance =
            DistanceCategory::get_distance(first_obstacle.y as f32, data.image_height); // Use perspective distance

        // Reason already includes direction relative to road start from center_line.rs
        write!(
            description,
            "Obstacle ({}) detected, {}",
            first_obstacle.reason, // e.g., "Potential gap near 11 o'clock direction"
            obstacle_distance      // e.g., "Near"
        )
        .unwrap();

        if data.obstacles.len() > 1 {
            write!(
                description,
                ", with {} other potential obstacles further down",
                data.obstacles.len() - 1
            )
            .unwrap();
        }

        Some(description)
    }
}
