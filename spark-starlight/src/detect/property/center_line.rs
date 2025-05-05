use crate::detect::constants::{
    MIN_CENTERLINE_POINTS_FOR_SHAPE, MIN_MASK_WIDTH_FOR_CENTER, NUM_VERTICAL_SAMPLES,
    OBSTACLE_GAP_ROWS_THRESHOLD, OBSTACLE_WIDTH_CHANGE_FACTOR, ROAD_CENTER_X_THRESHOLD,
    ROAD_START_Y_THRESHOLD, STRAIGHT_ROAD_X_DRIFT_THRESHOLD,
};
use crate::detect::property::direction::DirectionCategory;
use crate::detect::property::obstacle::ObstacleInfo;
use crate::detect::property::road_shape::RoadShape;
use bitvec::prelude::BitVec;
use log::error;
use std::ops::{Deref, DerefMut};

/// Center line is the line that runs through the center of the road mask.
/// It is used to analyze the road shape and detect obstacles.
#[derive(Debug, Clone)]
pub struct CenterLinePoint {
    pub y: u32,
    pub center_x: f32,
    pub width: u32,
}

#[derive(Debug, Clone)]
pub struct CenterLines(Vec<CenterLinePoint>);
impl Deref for CenterLines {
    type Target = Vec<CenterLinePoint>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for CenterLines {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl CenterLines {
    /// Extracts the center line and width profile of the road mask.
    pub fn extract_center_line(mask: &BitVec, image_width: u32, image_height: u32) -> Self {
        let mut center_line = CenterLines(Vec::new());
        if image_width == 0 || image_height == 0 || mask.is_empty() {
            return center_line;
        }
        if mask.len() != (image_width * image_height) as usize {
            error!(
                "Mask length mismatch: {} vs {}",
                mask.len(),
                image_width * image_height
            );
            return center_line;
        }

        let y_step = (image_height / (NUM_VERTICAL_SAMPLES + 1)).max(1); // Ensure step is at least 1

        for i in 1..=NUM_VERTICAL_SAMPLES {
            let y = image_height.saturating_sub(i * y_step); // Sample from bottom up
            if y >= image_height {
                continue;
            } // Should not happen with saturating_sub

            let row_start_index = (y * image_width) as usize;
            let mut min_x: Option<u32> = None;
            let mut max_x: Option<u32> = None;

            for x in 0..image_width {
                let index = row_start_index + x as usize;
                // Double check bounds just in case, though logic should prevent issues
                if index < mask.len() && mask[index] {
                    if min_x.is_none() {
                        min_x = Some(x);
                    }
                    max_x = Some(x);
                }
            }

            if let (Some(min_val), Some(max_val)) = (min_x, max_x) {
                let width = max_val.saturating_sub(min_val) + 1;
                if width >= MIN_MASK_WIDTH_FOR_CENTER {
                    let center_x = (min_val + max_val) as f32 / 2.0;
                    center_line.push(CenterLinePoint { y, center_x, width });
                }
            }
        }
        // Resulting points are ordered near (high y) to far (low y)
        center_line
    }

    /// Analyzes the shape (straight, curve left/right) based on center_line points.
    pub fn analyze_center_line_shape(&self, image_width: u32) -> RoadShape {
        if self.len() < MIN_CENTERLINE_POINTS_FOR_SHAPE {
            return RoadShape::Undetermined;
        }
        let nearest_point = &self[0];
        let farthest_point = self.last().unwrap();
        let normalized_drift =
            (farthest_point.center_x - nearest_point.center_x) / image_width as f32;

        if normalized_drift.abs() < STRAIGHT_ROAD_X_DRIFT_THRESHOLD {
            RoadShape::Straight
        } else if normalized_drift > 0.0 {
            // Farthest point is right -> Curves Right
            RoadShape::CurvesRight
        } else {
            // Farthest point is left -> Curves Left
            RoadShape::CurvesLeft
        }
    }

    /// Detects obstacles like gaps or significant narrowing based on the center line profile.
    /// Obstacle direction is calculated relative to the road's starting point.
    pub fn detect_obstacles(&self, image_width: u32, image_height: u32) -> Vec<ObstacleInfo> {
        let mut obstacles = Vec::new();
        if self.len() < 2 {
            return obstacles;
        }

        let y_step = (image_height / (NUM_VERTICAL_SAMPLES + 1)).max(1);
        let origin_point = &self[0]; // Use the nearest point on the centerline as the origin for obstacle directions

        // Detect Gaps
        let mut consecutive_missing_rows = 0;
        let mut last_valid_y = origin_point.y; // Keep track of the last row where the mask was found

        for i in 1..=NUM_VERTICAL_SAMPLES {
            let current_y = image_height.saturating_sub(i * y_step);
            let sampled_point_exists = self.iter().any(|p| p.y.abs_diff(current_y) < y_step / 2); // Check if a point exists near this y

            if sampled_point_exists {
                // Update last known valid y if we find the mask again
                if let Some(p) = self.iter().find(|p| p.y.abs_diff(current_y) < y_step / 2) {
                    last_valid_y = p.y;
                }
                consecutive_missing_rows = 0; // Reset gap counter
            } else {
                consecutive_missing_rows += 1;
                if consecutive_missing_rows == OBSTACLE_GAP_ROWS_THRESHOLD {
                    // Estimate gap position: halfway through the missing rows, below the last seen row
                    let gap_y_estimate =
                        last_valid_y.saturating_sub((OBSTACLE_GAP_ROWS_THRESHOLD * y_step) / 2);

                    // Find the center_x of the point just before the gap started
                    let point_before_gap = self.iter().find(|p| p.y == last_valid_y);
                    let center_x_estimate =
                        point_before_gap.map_or(origin_point.center_x, |p| p.center_x);

                    let obstacle_direction = DirectionCategory::get_direction(
                        center_x_estimate,
                        gap_y_estimate as f32,
                        Some((origin_point.center_x, origin_point.y as f32)), // Relative to road start
                        image_width,
                        image_height,
                    );

                    obstacles.push(ObstacleInfo {
                        y: gap_y_estimate,
                        center_x: center_x_estimate,
                        reason: format!("Potential gap near {}", obstacle_direction),
                    });
                    // Don't reset consecutive_missing_rows here, let it continue counting if the gap is larger
                }
            }
        }

        // Detect Narrowing
        for i in 0..(self.len() - 1) {
            let p_near = &self[i]; // Closer to viewer (higher y)
            let p_far = &self[i + 1]; // Further from viewer (lower y)

            // Check if width decreased significantly going further away
            if (p_far.width as f32) < (p_near.width as f32 * OBSTACLE_WIDTH_CHANGE_FACTOR) {
                // Check if this narrowing isn't already marked as part of a gap
                let y_mid_narrowing = (p_near.y + p_far.y) / 2;
                let is_near_gap = obstacles.iter().any(
                    |obs| {
                        obs.reason.contains("gap") && obs.y.abs_diff(y_mid_narrowing) < y_step * 2
                    }, // Generous window around gap Y
                );

                if !is_near_gap {
                    let obstacle_direction = DirectionCategory::get_direction(
                        p_far.center_x, // Use the position where narrowing is confirmed
                        p_far.y as f32,
                        Some((origin_point.center_x, origin_point.y as f32)), // Relative to road start
                        image_width,
                        image_height,
                    );
                    obstacles.push(ObstacleInfo {
                        y: p_far.y, // Position where the narrowing is significant
                        center_x: p_far.center_x,
                        reason: format!("Significant narrowing near {}", obstacle_direction),
                    });
                }
            }
        }

        // Sort obstacles by Y coordinate (descending: nearer first)
        obstacles.sort_by(|a, b| b.y.cmp(&a.y));

        obstacles
    }

    /// Checks if the road mask starts near the bottom edge of the image and is horizontally centered.
    /// Returns a tuple containing:
    /// - bool: whether the road starts at feet
    /// - Option<DirectionCategory>: the clock direction if not starting ideally at feet/center
    pub fn check_road_start(
        &self,
        image_width: u32,
        image_height: u32,
    ) -> (bool, Option<DirectionCategory>) {
        match self.first() {
            Some(nearest_point) => {
                // Vertical check: Is the nearest point close enough to the bottom?
                let normalized_y = nearest_point.y as f32 / image_height as f32;
                let is_at_bottom = normalized_y >= ROAD_START_Y_THRESHOLD;

                // Horizontal check: Is the nearest point reasonably centered horizontally?
                let normalized_x_offset = (nearest_point.center_x / image_width as f32 - 0.5).abs();
                let is_centered = normalized_x_offset <= ROAD_CENTER_X_THRESHOLD;

                if is_at_bottom && is_centered {
                    (true, None) // Ideal start: at feet and centered
                } else {
                    // Not ideal start, calculate direction relative to image bottom-center
                    let direction = DirectionCategory::get_direction(
                        nearest_point.center_x,
                        nearest_point.y as f32,
                        None, // Use default image bottom-center origin
                        image_width,
                        image_height,
                    );
                    (false, Some(direction))
                }
            }
            None => (false, None), // No centerline points found
        }
    }
}
