use anyhow::Result;
use rayon::prelude::*;
use spark_ffmpeg::util::ptr_wrapper::SafePtr;
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
        let tensor_ptr = SafePtr::new(tensor.as_mut_ptr());

        self
            .raw_data()?
            .iter()
            .enumerate()
            .par_bridge()
            .for_each(|(index, &value)| {
                let index = match index % 3 {
                    0 => index / 3,
                    1 => size / 3 + index / 3,
                    2 => size * 2 / 3 + index / 3,
                    _ => panic!("Invalid index occur when extract image to tensor."),
                };

                unsafe {
                    tensor_ptr.add(index).write(value as f32 / 255.);
                }
            });

        Ok(tensor)
    }
}