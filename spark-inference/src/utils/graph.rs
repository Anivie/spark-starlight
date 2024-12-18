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
pub enum SamPrompt<T: Num> {
    Box(Box<T>),
    Point(Point<T>),
    Both(Point<T>, Box<T>),
}

impl<T: Num> SamPrompt<T> {
    pub fn point(x: T, y: T) -> SamPrompt<T> {
        let point = Point { x, y };
        SamPrompt::Point(point)
    }

    pub fn boxes(x: T, y: T, width: T, height: T) -> SamPrompt<T> {
        let boxes = Box {
            x,
            y,
            width,
            height,
        };
        SamPrompt::Box(boxes)
    }

    pub fn both(boxes: (T, T, T, T), point: (T, T)) -> SamPrompt<T> {
        let boxes = Box {
            x: boxes.0,
            y: boxes.1,
            width: boxes.2,
            height: boxes.3,
        };
        let point = Point {
            x: point.0,
            y: point.1,
        };
        SamPrompt::Both(point, boxes)
    }
}
