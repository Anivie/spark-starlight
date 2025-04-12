use crate::detect::analysis::Describer;
use crate::detect::property::analyse_result::RoadAnalysisData;
use crate::detect::property::road_shape::RoadShape;
use std::fmt::Write;

#[derive(Debug, Copy, Clone)]
pub struct RoadShapeDescriber;

impl Describer for RoadShapeDescriber {
    async fn describe(
        &self,
        data: &RoadAnalysisData<'_>,
        _object_type_name: &str,
    ) -> Option<String> {
        if data.center_lines.is_empty() {
            return None;
        }

        let mut description = String::new();
        match data.shape {
            RoadShape::Straight => write!(description, "It proceeds straight").unwrap(),
            RoadShape::CurvesLeft => write!(description, "It curves left").unwrap(),
            RoadShape::CurvesRight => write!(description, "It curves right").unwrap(),
            RoadShape::Undetermined => write!(description, "Its shape is unclear").unwrap(),
        }
        Some(description)
    }
}
