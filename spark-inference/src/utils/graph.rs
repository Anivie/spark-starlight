use bitvec::macros::internal::funty::Numeric;

#[derive(Debug)]
pub struct Point<T: Numeric> {
    pub x: T,
    pub y: T,
}