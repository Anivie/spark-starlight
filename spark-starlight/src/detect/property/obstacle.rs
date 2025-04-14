#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ObstacleInfo {
    pub y: u32,
    pub center_x: f32,
    pub reason: String, // Includes direction description now
}
