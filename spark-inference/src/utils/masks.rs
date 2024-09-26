use crate::utils::ptr_wrapper::SafeVecPtr;
use anyhow::Result;
use bitvec::vec::BitVec;
use rayon::prelude::*;
use spark_ffmpeg::pixel::pixel_formater::RGB;
use spark_media::Image;

pub trait ApplyMask {
    fn layering_mask(&mut self, dim_index: usize, mask: &BitVec, apply_color: RGB) -> Result<()>;
}

impl ApplyMask for Image {
    fn layering_mask(&mut self, dim_index: usize, mask: &BitVec, apply_color: RGB) -> Result<()> {
        let frame = self.frame_mut()?;
        let mut data = frame.get_raw_data(dim_index);
        let ptr = SafeVecPtr::new(data.as_mut_ptr());

        (0 .. (frame.get_width() * frame.get_height() * 3) as usize)
            .into_par_iter()
            .for_each(|index| {
                if !unsafe {
                    *mask.get_unchecked(index / 3)
                } {
                    return;
                }

                match (index % 3, apply_color.0 > 0, apply_color.1 > 0, apply_color.2 > 0) {
                    (0, true, _, _) => unsafe {
                        let after = (*ptr.add(index)).saturating_add(apply_color.0);
                        ptr.add(index).write(after);
                    }
                    (1, _, true, _) => unsafe {
                        let after = (*ptr.add(index)).saturating_add(apply_color.1);
                        ptr.add(index).write(after);
                    }
                    (2, _, _, true) => unsafe {
                        let after = (*ptr.add(index)).saturating_add(apply_color.2);
                        ptr.add(index).write(after);
                    }
                    _ => {}
                }
            });

        drop(frame);
        self.refresh_packet()?;

        Ok(())
    }
}