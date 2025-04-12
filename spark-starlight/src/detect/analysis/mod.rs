pub mod compose;
mod object_detected_describer;
mod obstacle_describer;
mod path_ending_describer;
mod road_shape_describer;
mod road_start_describer;
mod user_position_describer;

use crate::detect::property::analyse_result::RoadAnalysisData;

/// Trait for generating a specific *part* of the textual description based on analysis results.
pub trait Describer {
    /// Generates a specific part of the description based on the analysis data.
    /// Returns Some(description_part) if relevant, None otherwise.
    fn describe(&self, data: &RoadAnalysisData, object_type_name: &str) -> Option<String>;
}
