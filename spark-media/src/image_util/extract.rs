use std::ops::Deref;
use spark_ffmpeg::avframe::AVFrame;
use anyhow::Result;
use rayon::prelude::*;

struct SafeVecPtr(*mut f32);
impl Deref for SafeVecPtr {
    type Target = *mut f32;

    fn deref(&self) -> &Self::Target {
        unsafe { &self.0 }
    }
}
unsafe impl Send for SafeVecPtr {}
unsafe impl Sync for SafeVecPtr {}

pub trait ExtraToTensor {
    fn extra_standard_image_to_tensor(&self) -> Result<Vec<f32>>;
}

impl ExtraToTensor for AVFrame {
    fn extra_standard_image_to_tensor(&self) -> Result<Vec<f32>> {
        let size = (self.get_width() * self.get_width() * 3) as usize;

        let mut tensor = {
            let mut vec = Vec::with_capacity(size);
            unsafe { vec.set_len(size); }
            vec
        };
        let tensor_ptr = SafeVecPtr(tensor.as_mut_ptr());

        self
            .get_data(0)
            .iter()
            .enumerate()
            .par_bridge()
            .for_each(|(index, &value)| {
                if index % 3 == 0 {
                    unsafe { *tensor_ptr.wrapping_add(index / 3) = value as f32 / 255.; }
                }

                if index % 3 == 1 {
                    unsafe { *tensor_ptr.wrapping_add(size / 3 + index / 3) = value as f32 / 255.; }
                }

                if index % 3 == 2 {
                    unsafe { *tensor_ptr.wrapping_add(size * 2 / 3 + index / 3) = value as f32 / 255.; }
                }
            });

        Ok(tensor)
    }
}