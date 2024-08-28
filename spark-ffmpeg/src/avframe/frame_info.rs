use crate::avframe::AVFrame;

impl AVFrame {
    pub fn get_width(&self) -> i32 {
        unsafe { (*self.inner).width }
    }

    pub fn get_height(&self) -> i32 {
        unsafe { (*self.inner).height }
    }
}