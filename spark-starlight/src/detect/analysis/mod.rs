pub mod compose;
mod object_detected_describer;
mod obstacle_describer;
mod path_ending_describer;
mod road_shape_describer;
mod road_start_describer;
mod user_position_describer;
#[macro_use]
mod dispatch_macro;

use crate::detect::analysis::object_detected_describer::DetectedObjectDescriber;
use crate::detect::analysis::obstacle_describer::ObstacleDescriber;
use crate::detect::analysis::path_ending_describer::PathEndingDescriber;
use crate::detect::analysis::road_shape_describer::RoadShapeDescriber;
use crate::detect::analysis::road_start_describer::RoadStartDescriber;
use crate::detect::analysis::user_position_describer::UserPositionDescriber;
use crate::detect::property::analyse_result::RoadAnalysisData;

/// Trait for generating a specific *part* of the textual description based on analysis results.
pub trait Describer {
    /// Generates a specific part of the description based on the analysis data.
    /// Returns Some(description_part) if relevant, None otherwise.
    async fn describe(&self, data: &RoadAnalysisData, object_type_name: &str) -> Option<String>;
}

define_describer![
    ObjectDetected => DetectedObjectDescriber,
    Obstacle => ObstacleDescriber,
    PathEnding => PathEndingDescriber,
    RoadShape => RoadShapeDescriber,
    RoadStart => RoadStartDescriber,
    UserPosition => UserPositionDescriber,
];
