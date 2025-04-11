pub mod anchor;
pub mod mask;
pub(crate) mod property;

pub(crate) const NUM_VERTICAL_SAMPLES: u32 = 20;
pub(crate) const MIN_MASK_WIDTH_FOR_CENTER: u32 = 5;
pub(crate) const ROAD_START_Y_THRESHOLD: f32 = 0.90; // Normalized Y threshold for "at feet"
pub(crate) const ROAD_CENTER_X_THRESHOLD: f32 = 0.2; // Normalized X offset tolerance for "centered"
pub(crate) const MIN_CENTERLINE_POINTS_FOR_SHAPE: usize = 5;
pub(crate) const STRAIGHT_ROAD_X_DRIFT_THRESHOLD: f32 = 0.05; // Normalized X drift for straight road
pub(crate) const OBSTACLE_WIDTH_CHANGE_FACTOR: f32 = 0.5;
pub(crate) const OBSTACLE_GAP_ROWS_THRESHOLD: u32 = 3;

// --- Perspective Correction Constants ---
/// Power factor for perspective correction in distance calculation.
/// Values < 1.0 make objects near the bottom seem closer (steeper distance falloff).
/// Values > 1.0 make objects near the bottom seem relatively farther (gentler distance falloff).
/// 1.0 is linear (no perspective correction).
/// Tunable value, start around 0.6 - 0.8.
pub(crate) const DISTANCE_PERSPECTIVE_POWER: f32 = 0.7;

/// Threshold for height ratio (box_height / image_height) to consider an object "covering path" when very near.
pub(crate) const NEAR_OBJECT_HEIGHT_THRESHOLD_FACTOR: f32 = 0.3; // 30% of image height
/// Threshold for width ratio (box_width / image_width) to consider an object "covering path" when very near.
pub(crate) const NEAR_OBJECT_WIDTH_THRESHOLD_FACTOR: f32 = 0.5; // 50% of image width
