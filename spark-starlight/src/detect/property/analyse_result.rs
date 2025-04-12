use crate::detect::property::center_line::CenterLines;
use crate::detect::property::direction::DirectionCategory;
use crate::detect::property::obstacle::ObstacleInfo;
use crate::detect::property::road_shape::RoadShape;
use spark_inference::inference::yolo::inference_yolo_detect::YoloDetectResult;

#[derive(Debug, Clone)]
pub struct RoadAnalysisData<'a> {
    // Renamed to avoid confusion with the final *output* struct if needed later
    pub image_width: u32,
    pub image_height: u32,
    pub detect_results: &'a [YoloDetectResult], // Renamed for clarity

    pub shape: RoadShape,
    pub obstacles: Vec<ObstacleInfo>,
    pub center_lines: CenterLines,

    // These are essential intermediate results derived from center_lines
    pub starts_at_feet: bool,
    pub start_direction: Option<DirectionCategory>, // Direction if not starting ideally
}
