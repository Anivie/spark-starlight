use std::mem::ManuallyDrop;
use iter::frame_iter::PixelData;

pub mod open_frame;
pub mod frame_info;
mod iter;

wrap!(
  AVFrame drop2 av_frame_free
);

impl AVFrame {
  pub fn get_raw_data(&self, index: usize) -> ManuallyDrop<Vec<u8>> {
    let size = (self.linesize[index] * self.height) as usize;
    unsafe {
      ManuallyDrop::new(Vec::from_raw_parts(self.data[index], size, size))
    }
  }

  pub fn get_rgb_data(&self, index: usize) -> PixelData {
    let size = (self.linesize[index] * self.height) as usize;
    let data = unsafe {
      Vec::from_raw_parts(self.data[index], size, size)
    };
    PixelData {
        inner: ManuallyDrop::new(data),
        width: self.width,
        height: self.height,
        line_size: self.linesize[index],
        current_x: 0,
        current_y: 0,
    }
  }
}