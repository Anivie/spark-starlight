use bitvec::macros::internal::funty::Numeric;

#[derive(Debug)]
pub struct Point<T: Numeric> {
    pub x: T,
    pub y: T,
}

#[derive(Debug)]
pub struct Box<T: Numeric> {
    pub x: T,
    pub y: T,
    pub width: T,
    pub height: T,
}

pub enum BoxOrPoint<T: Numeric> {
    Box(Box<T>),
    Point(Point<T>),
}

impl<T: Numeric> BoxOrPoint<T> {
    pub fn point(x: T, y: T) -> BoxOrPoint<T> {
        let point = Point {
            x,
            y,
        };
        BoxOrPoint::Point(point)
    }

    pub fn boxes(x: T, y: T, width: T, height: T) -> BoxOrPoint<T> {
        let boxes = Box {
            x,
            y,
            width,
            height
        };
        BoxOrPoint::Box(boxes)
    }
}