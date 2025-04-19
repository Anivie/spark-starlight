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

#[derive(Clone)]
pub enum SamPrompt<T: Num> {
    Box(Box<T>),
    Point(Point<T>),
    Points(Vec<Point<T>>),
    Both(Point<T>, Box<T>),
}
