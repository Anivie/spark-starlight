use num::Num;

#[derive(Copy, Clone, Debug)]
pub struct Point<T: Num> {
    pub x: T,
    pub y: T,
}

#[derive(Copy, Clone, Debug)]
pub struct Box<T: Num> {
    pub x: T,
    pub y: T,
    pub width: T,
    pub height: T,
}

#[derive(Copy, Clone)]
pub enum BoxOrPoint<T: Num> {
    Box(Box<T>),
    Point(Point<T>),
}

impl<T: Num> BoxOrPoint<T> {
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