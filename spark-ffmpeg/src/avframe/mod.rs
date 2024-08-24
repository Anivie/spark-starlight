use std::ffi::c_void;
use crate::ffi::av_freep;

mod open_frame;

#[derive(Debug, Clone)]
struct MarkAlloc(*mut c_void);
impl Drop for MarkAlloc {
    fn drop(&mut self) {
      unsafe {
        av_freep(self.0);
      }
    }
}

wrap!(
  AVFrame {
    is_alloc_image: Option<MarkAlloc>,
  } drop2 av_frame_free
);