use anyhow::Result;
use bitvec::macros::internal::funty::Fundamental;
use bitvec::vec::BitVec;
use rayon::prelude::*;
use spark_ffmpeg::pixel::pixel_formater::RGB;
use spark_media::Image;
use crate::utils::SafeVecPtr;

pub trait ApplyMask {
    fn layering_mask(&mut self, dim_index: usize, mask: &BitVec, apply_color: RGB) -> Result<()>;
}

impl ApplyMask for Image {
    fn layering_mask(&mut self, dim_index: usize, mask: &BitVec, apply_color: RGB) -> Result<()> {
        let frame = self.frame_mut()?;
        let mut data = frame.get_raw_data(dim_index);
        let ptr = SafeVecPtr(data.as_mut_ptr());

        (0 .. (frame.get_width() * frame.get_height())as usize)
            .into_par_iter()
            .for_each(|index| {
                let mask = unsafe {
                    mask.get_unchecked(index)
                };
                if index % 3 == 0 {
                    if mask.as_bool() {
                        unsafe {
                            ptr.add(index).write((*ptr.0).saturating_add(apply_color.0));
                        }
                    }
                }
                if index % 3 == 1 {
                    if mask.as_bool() {
                        unsafe {
                            ptr.add(index).write((*ptr.0).saturating_add(apply_color.0));
                        }
                    }
                }
                if index % 3 == 2 {
                    if mask.as_bool() {
                        unsafe {
                            ptr.add(index).write((*ptr.0).saturating_add(apply_color.0));
                        }
                    }
                }
            });

        Ok(())
    }
}