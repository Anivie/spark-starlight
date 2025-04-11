use crate::detect::property::direction::DirectionCategory;
use crate::detect::property::distance::DistanceCategory;
use crate::detect::{NEAR_OBJECT_HEIGHT_THRESHOLD_FACTOR, NEAR_OBJECT_WIDTH_THRESHOLD_FACTOR};
use log::error;
use spark_inference::inference::yolo::inference_yolo_detect::YoloDetectResult;

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
        let distance = DistanceCategory::get_distance(result.y, image_height);
        let direction =
            DirectionCategory::get_direction(result.x, result.y, None, image_width, image_height);

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
