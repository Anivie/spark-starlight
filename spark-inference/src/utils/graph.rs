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
}
