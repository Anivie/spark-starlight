use crate::avframe::AVFrame;

#[derive(Debug)]
pub struct RGB(pub u8, pub u8, pub u8);

#[derive(Debug, Clone)]
pub struct PixelIterator<'a> {
    pub(super) current_x: usize,
    pub(super) current_y: usize,

    pub(super) frame: &'a AVFrame,
}

impl Iterator for PixelIterator<'_> {
    type Item = (u32, u32, RGB);

    fn next(&mut self) -> Option<Self::Item> {
        let rgb = {
            let level_surface = &self.frame.data[0];
            let line_size = self.frame.linesize[0] as usize;
            unsafe {
                RGB(
                    *level_surface.add(self.current_y * line_size + self.current_x * 3),
                    *level_surface.add(self.current_y * line_size + self.current_x * 3 + 1),
                    *level_surface.add(self.current_y * line_size + self.current_x * 3 + 2),
                )
            }
        };

        self.current_x += 1;
        if self.current_x >= self.frame.width as usize {
            self.current_x = 0;
            self.current_y += 1;
        }

        if self.current_y >= self.frame.height as usize {
            None
        }else {
            Some((self.current_x as u32, self.current_y as u32, rgb))
        }
    }
}