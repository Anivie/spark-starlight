use std::mem::ManuallyDrop;

pub struct PixelData {
    pub(in crate::avframe) inner: ManuallyDrop<Vec<u8>>,
    pub(in crate::avframe) width: i32,
    pub(in crate::avframe) height: i32,
    pub(in crate::avframe) line_size: i32,

    pub(in crate::avframe) current_x: i32,
    pub(in crate::avframe) current_y: i32,
}

#[derive(Debug, Copy, Clone)]
pub struct RGB(pub u8, pub u8, pub u8);

impl Iterator for PixelData {
    type Item = (i32, i32, RGB);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_x >= self.width {
            self.current_x = 0;
            self.current_y += 1;
        }

        if self.current_y >= self.height {
            return None;
        }

        let index = (self.current_y * self.line_size + self.current_x * 3) as usize;
        let r = self.inner[index];
        let g = self.inner[index + 1];
        let b = self.inner[index + 2];

        self.current_x += 1;

        Some((self.current_x, self.current_y, RGB(r, g, b)))
    }
}
