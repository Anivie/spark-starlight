use crate::utils::ptr_wrapper::SafeVecPtr;
use anyhow::Result;
use rayon::prelude::*;
use spark_media::Image;

pub trait ExtraToTensor {
    fn extra_standard_image_to_tensor(&self) -> Result<Vec<f32>>;
}

impl ExtraToTensor for Image {
    fn extra_standard_image_to_tensor(&self) -> Result<Vec<f32>> {
        let size = (self.get_width() * self.get_width() * 3) as usize;

        let mut tensor: Vec<f32> = {
            let mut vec = Vec::with_capacity(size);
            unsafe { vec.set_len(size); }
            vec
        };
        let tensor_ptr = SafeVecPtr::new(tensor.as_mut_ptr());

        self
            .frame()?
            .get_raw_data(0)
            .iter()
            .enumerate()
            .par_bridge()
            .for_each(|(index, &value)| {
                if index % 3 == 0 {
                    unsafe { tensor_ptr.add(index / 3).write(value as f32 / 255.) }
                }

                if index % 3 == 1 {
                    unsafe { tensor_ptr.add(size / 3 + index / 3).write(value as f32 / 255.); }
                }

                if index % 3 == 2 {
                    unsafe { tensor_ptr.add(size * 2 / 3 + index / 3).write(value as f32 / 255.); }
                }
            });

        Ok(tensor)
    }
}