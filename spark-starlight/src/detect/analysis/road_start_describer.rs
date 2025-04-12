use crate::detect::analysis::Describer;
use crate::detect::property::analyse_result::RoadAnalysisData;
use crate::detect::property::distance::DistanceCategory;
use std::fmt::Write;

#[derive(Debug, Copy, Clone)]
pub struct RoadStartDescriber;

impl Describer for RoadStartDescriber {
    async fn describe(
        &self,
        data: &RoadAnalysisData<'_>,
        object_type_name: &str,
    ) -> Option<String> {
        let mut description = String::new();

        if data.center_lines.is_empty() {
            // If centerline failed but objects were detected, CompositeDescriber might still call other describers.
            // Let's provide a specific message here if *nothing* at all was found.
            if data.detect_results.is_empty() {
                return Some(format!(
                    "No {} or related objects detected.",
                    object_type_name
                ));
            } else {
                // Let the object describer handle the detected objects.
                // Optionally, add a note about the missing road context.
                // return Some(format!("Could not clearly identify the {}. ", object_type_name));
                // Returning None here might be cleaner, letting the object describer start the sentence.
                return None;
            }
        } else if data.starts_at_feet {
            write!(
                description,
                "The {} starts at your feet.", // Removed trailing space, will be added by composer
                object_type_name
            )
            .unwrap();
            Some(description)
        } else {
            // Use perspective-corrected distance for the warning message
            if let Some(nearest_point) = data.center_lines.first() {
                let start_distance =
                    DistanceCategory::get_distance(nearest_point.y as f32, data.image_height);
                let direction_str = match &data.start_direction {
                    Some(dir) => format!("{}", dir),
                    None => "centered".to_string(), // Should have direction if not starts_at_feet
                };
                write!(
                    description,
                    "Warning: The {} doesn't start directly underfoot. It appears {} in the {}.",
                    object_type_name,
                    start_distance, // e.g., "Relatively near", "Near"
                    direction_str
                )
                .unwrap();
                Some(description)
            } else {
                // This case is covered by center_lines.is_empty()
                None
            }
        }
    }
}
