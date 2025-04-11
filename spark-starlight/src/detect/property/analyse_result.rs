use crate::detect::property::obstacle::ObstacleInfo;
use crate::detect::property::road_shape::RoadShape;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone)]
pub struct RoadAnalysisResult {
    pub starts_at_feet: bool,
    pub shape: RoadShape,
    pub obstacles: Vec<ObstacleInfo>,
    pub description: String,
}

impl Display for RoadAnalysisResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Starts at feet: {}, Shape: {:?}, Obstacles: {:?}, Description: {}",
            self.starts_at_feet, self.shape, self.obstacles, self.description
        )
    }
}
