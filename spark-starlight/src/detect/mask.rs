use crate::detect::analysis::compose::CompositeDescriber;
use crate::detect::property::analyse_result::RoadAnalysisData;
use crate::detect::property::center_line::CenterLines;
use crate::detect::property::road_shape::RoadShape;
use bitvec::prelude::BitVec;
use log::{error, info};
use spark_inference::inference::yolo::inference_yolo_detect::YoloDetectResult;

pub async fn analyze_road_mask(
    mask: &BitVec,
    detections: &[YoloDetectResult],
    image_width: u32,
    image_height: u32,
    object_type_name: &str,
) -> String {
    let analysis_data_option = perform_core_analysis(mask, detections, image_width, image_height);

    let description = match analysis_data_option {
        Some(data) => {
            // Use the CompositeDescriber to generate the final text
            let composite_describer = CompositeDescriber::new();
            composite_describer.describe(&data, object_type_name).await // Call the composite describe method
        }
        None => {
            // Core analysis failed, provide a generic failure message
            format!(
                "Analysis failed for {}: Invalid input data or no road features found.",
                object_type_name
            )
        }
    };

    description
}

// --- Move perform_core_analysis here if not in analysis/mod.rs ---
/// Analyzes a road mask to determine shape, obstacles, and starting position.
/// Args:
///    mask (BitVec): The road mask to analyze.
///    image_width (u32): The width of the image frame.
///    image_height (u32): The height of the image frame.
///    object_type_name (str): The name of the object type (e.g., "Highway", "Sidewalk").
///
/// Returns:
///   Option<RoadAnalysisData>: A struct containing analysis results or None if basic checks fail.
fn perform_core_analysis<'a>(
    mask: &'a BitVec,
    detections: &'a [YoloDetectResult],
    image_width: u32,
    image_height: u32,
) -> Option<RoadAnalysisData<'a>> {
    if image_width == 0
        || image_height == 0
        || mask.is_empty()
        || mask.len() != (image_width * image_height) as usize
    {
        error!("Invalid input mask or dimensions for core analysis.");
        return None;
    }

    // 1. Calculate Centerline
    // Make sure CenterLines and its methods are accessible (e.g., pub in property module)
    let center_lines = CenterLines::extract_center_line(mask, image_width, image_height);

    // If centerline extraction fails completely, we might still have detected objects.
    // Return a RoadAnalysisData with centerline-dependent fields empty/defaulted.
    if center_lines.is_empty() {
        info!("No significant centerline found in the mask. Proceeding with object detection analysis only.");
        // Return data structure allowing object description even without road context
        return Some(RoadAnalysisData {
            image_width,
            image_height,
            detect_results: detections,
            shape: RoadShape::Undetermined, // No centerline, shape unknown
            obstacles: vec![],              // No centerline, no obstacles derived from it
            center_lines,                   // Empty centerline vector
            starts_at_feet: false,          // Cannot start at feet without centerline
            start_direction: None,
        });
    }

    // 2. Analyze Shape
    let shape = center_lines.analyze_center_line_shape(image_width);

    // 3. Detect Obstacles
    let obstacles = center_lines.detect_obstacles(image_width, image_height);

    // 4. Check Start Position
    let (starts_at_feet, start_direction) =
        center_lines.check_road_start(image_width, image_height);

    // 5. Assemble Intermediate Data Structure
    Some(RoadAnalysisData {
        image_width,
        image_height,
        detect_results: detections,
        shape,
        obstacles,
        center_lines, // Move the calculated centerline here
        starts_at_feet,
        start_direction,
    })
}

pub fn get_best_highway(masks: &Vec<BitVec>) -> Option<&BitVec> {
    masks.iter().max_by_key(|mask| {
        let center_line = CenterLines::extract_center_line(mask, 1024, 1024);
        center_line.first().map_or(0, |p| p.y) // Use y of nearest point
    })
}
