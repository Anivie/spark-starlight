use crate::detect::analysis::Describer;
use crate::detect::constants::{
    NEAR_OBJECT_HEIGHT_THRESHOLD_FACTOR, NEAR_OBJECT_WIDTH_THRESHOLD_FACTOR,
};
use crate::detect::property::analyse_result::RoadAnalysisData;
use crate::detect::property::direction::DirectionCategory;
use crate::detect::property::distance::DistanceCategory;
use std::fmt::Write;

pub struct DetectedObjectDescriber;

impl Describer for DetectedObjectDescriber {
    fn describe(&self, data: &RoadAnalysisData, _object_type_name: &str) -> Option<String> {
        if data.detect_results.is_empty() {
            // Only add "No specific objects detected" if the centerline also failed.
            // If centerline exists, the lack of objects is implicitly covered.
            if data.center_lines.is_empty() {
                // Check if RoadStartDescriber already handled the "No road or objects" case
                // This describer shouldn't repeat that. Let's return None here.
                // The CompositeDescriber can handle the "nothing found at all" scenario.
                return None;
            } else {
                return None; // No objects to describe
            }
        }

        let mut description = String::new();
        let mut object_descs = Vec::new();

        if data.center_lines.is_empty() {
            // If no road context, just list objects
            write!(
                description,
                "Detected {} objects: ",
                data.detect_results.len()
            )
            .unwrap();
        } else {
            // If road context exists, phrase it as objects *on* or *near* the road
            write!(description, "Additionally, detected objects include: ").unwrap();
        }

        for (i, result) in data.detect_results.iter().enumerate() {
            // Calculate perspective-corrected distance and direction relative to image bottom-center
            let distance = DistanceCategory::get_distance(result.y, data.image_height);
            let direction = DirectionCategory::get_direction(
                result.x,
                result.y,
                None, // Relative to bottom-center origin
                data.image_width,
                data.image_height,
            );

            let object_desc: String;

            // Perspective Adjustment for "Very Near" Objects
            if distance == DistanceCategory::VeryNear {
                let box_height_ratio = result.height / data.image_height as f32;
                let box_width_ratio = result.width / data.image_width as f32;

                if box_height_ratio >= NEAR_OBJECT_HEIGHT_THRESHOLD_FACTOR
                    && box_width_ratio >= NEAR_OBJECT_WIDTH_THRESHOLD_FACTOR
                {
                    // Special description for large, close objects that might be the road surface itself
                    // Using a generic term might be safer than assuming it's `object_type_name`
                    object_desc = format!(
                        "a large surface covering the path underfoot, primarily in the {}",
                        direction
                    );
                    // Avoid numbering if it's likely the main surface
                } else {
                    // Standard description for objects that are "Very near" but not huge
                    object_desc = format!(
                        "Item {} ({}, {})",
                        i + 1,
                        distance, // "Very near"
                        direction
                    );
                }
            } else {
                // Standard description for objects not classified as "Very near"
                object_desc = format!(
                    "Item {} ({}, {})",
                    i + 1,
                    distance, // Near, Far, etc.
                    direction
                );
            }
            object_descs.push(object_desc);
        }
        write!(description, "{}", object_descs.join("; ")).unwrap(); // Join with semicolon

        Some(description)
    }
}
