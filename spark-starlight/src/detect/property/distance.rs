use crate::detect::DISTANCE_PERSPECTIVE_POWER;
use log::error;
use std::fmt::{Display, Formatter};

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum DistanceCategory {
    VeryNear,       // e.g., At your feet / Underfoot
    RelativelyNear, // Close
    Near,           // Mid-range
    Far,            // Distant
    Unknown,        // Error case
}

impl Display for DistanceCategory {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            DistanceCategory::VeryNear => write!(f, "Very near"), // Base description
            DistanceCategory::RelativelyNear => write!(f, "Relatively near"),
            DistanceCategory::Near => write!(f, "Near"),
            DistanceCategory::Far => write!(f, "Far"),
            DistanceCategory::Unknown => write!(f, "Unknown distance"),
        }
    }
}

impl DistanceCategory {
    /// Calculates the perceived distance of an anchor point based on its y-coordinate.
    /// Considers perspective (lower y in image means closer to the viewer).
    /// Applies a non-linear mapping to account for perspective distortion.
    ///
    /// Args:
    ///     y (f32): The y-coordinate of the anchor point (center of the bounding box).
    ///     image_height (u32): The total height of the image frame in pixels.
    ///
    /// Returns:
    ///     DistanceCategory: The perceived distance category.
    pub fn get_distance(y: f32, image_height: u32) -> Self {
        if image_height == 0 {
            error!("Image height cannot be zero for distance calculation.");
            return DistanceCategory::Unknown;
        }

        // Normalize y coordinate to the range [0.0, 1.0]
        // 0.0 is the top of the image (far), 1.0 is the bottom (near).
        let normalized_y = (y / image_height as f32).max(0.0).min(1.0);

        // Apply perspective correction using a power function.
        // This maps the linear normalized_y to a non-linear scale.
        // With power < 1, values closer to 1.0 (bottom) increase faster,
        // simulating the visual effect of things getting much closer rapidly at the bottom.
        let perspective_adjusted_y = normalized_y.powf(DISTANCE_PERSPECTIVE_POWER);

        // Define thresholds based on the perspective-adjusted value.
        // *** These thresholds likely need tuning based on the chosen power factor ***
        // *** and the desired feel of "near" vs "far". Experimentation is key! ***
        if perspective_adjusted_y > 0.90 {
            // Adjusted from 0.85
            DistanceCategory::VeryNear
        } else if perspective_adjusted_y > 0.75 {
            // Adjusted from 0.60
            DistanceCategory::RelativelyNear
        } else if perspective_adjusted_y > 0.50 {
            // Adjusted from 0.35
            DistanceCategory::Near
        } else {
            DistanceCategory::Far
        }
    }
}
