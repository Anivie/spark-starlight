use std::mem::ManuallyDrop;

pub mod open_frame;
pub mod frame_info;

wrap!(
  AVFrame drop2 av_frame_free
);

impl AVFrame {
  pub fn get_data(&self, index: usize) -> ManuallyDrop<Vec<u8>> {
    let size = self.linesize[index] * self.height;
    unsafe {
      ManuallyDrop::new(Vec::from_raw_parts(self.data[index], size as usize, size as usize))
    }
  }
}