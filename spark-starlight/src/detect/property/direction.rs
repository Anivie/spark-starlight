use log::error;
use std::f32::consts::PI;
use std::fmt::{Display, Formatter};

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum DirectionCategory {
    Clock(&'static str),
    Unknown,
}

impl Display for DirectionCategory {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            DirectionCategory::Clock(s) => write!(f, "{} direction", s),
            DirectionCategory::Unknown => write!(f, "Unknown direction"),
        }
    }
}

impl DirectionCategory {
    /// Calculates the perceived direction of a point based on its angle relative
    /// to an origin point (defaults to center bottom of the frame).
    ///
    /// Args:
    ///     x (f32): The x-coordinate of the target point.
    ///     y (f32): The y-coordinate of the target point (0 is top, image_height is bottom).
    ///     origin (Option<(f32, f32)>): The (x, y) origin for angle calculation. If None, uses image bottom-center.
    ///     image_width (u32): The total width of the image frame in pixels.
    ///     image_height (u32): The total height of the image frame in pixels.
    ///
    /// Returns:
    ///     DirectionCategory: The direction represented as a clock face position (9 to 3 o'clock).
    pub fn get_direction(
        x: f32,
        y: f32,
        origin: Option<(f32, f32)>,
        image_width: u32,
        image_height: u32,
    ) -> DirectionCategory {
        if image_width == 0 || image_height == 0 {
            error!("Image dimensions cannot be zero for direction calculation.");
            return DirectionCategory::Unknown;
        }

        // Define the origin: Use provided origin or default to center bottom.
        let (origin_x, origin_y) =
            origin.unwrap_or((image_width as f32 / 2.0, image_height as f32));

        let dx = x - origin_x;
        // dy is calculated such that positive values mean "up" from the origin towards the top of the image.
        let dy = origin_y - y;

        // Handle edge cases where dy is very close to zero or slightly negative
        // atan2 handles dy=0 correctly, mapping to +/- PI/2 (3/9 o'clock).
        // atan2(0, 0) -> 0 (12 o'clock).
        if dy < 0.0 && dy.abs() < 1e-6 {
            // Effectively on the origin's horizontal line, treat dy as 0 for atan2.
        } else if dy < 0.0 {
            // Point is below the origin line. This can happen if origin is not bottom of image (e.g., road mask origin).
            // The angle calculation might be less intuitive ("behind"), but atan2 handles it.
            // We clamp later to the forward arc.
            // warn!( "Point y ({}) is below the origin line y ({}), direction might be less intuitive.", y, origin_y);
        }

        // Calculate the angle using atan2(dx, dy).
        let angle_rad = dx.atan2(dy); // Range: -PI to PI

        // Clamp the angle to the forward arc [-PI/2, PI/2] (9 o'clock to 3 o'clock).
        // This ensures we only describe directions in front of the origin.
        let clamped_angle_rad = angle_rad.max(-PI / 2.0).min(PI / 2.0);

        // Map the angle [-PI/2, PI/2] to an index [0, 12] for clock strings.
        let index = (((clamped_angle_rad + PI / 2.0) / PI) * 12.0).round() as usize;
        let index = index.min(12); // Ensure index is within bounds [0, 12]

        let clock_str = match index {
            0 => "9 o'clock",
            1 => "9:30",
            2 => "10 o'clock",
            3 => "10:30",
            4 => "11 o'clock",
            5 => "11:30",
            6 => "12 o'clock",
            7 => "12:30",
            8 => "1 o'clock",
            9 => "1:30",
            10 => "2 o'clock",
            11 => "2:30",
            12 => "3 o'clock",
            _ => unreachable!(),
        };

        DirectionCategory::Clock(clock_str)
    }
}
