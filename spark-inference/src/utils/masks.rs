use anyhow::Result;
use bitvec::vec::BitVec;
use rayon::prelude::*;
use spark_ffmpeg::pixel::pixel_formater::RGB;
use spark_ffmpeg::util::ptr_wrapper::SafePtr;
use spark_media::Image;

pub trait ApplyMask {
    fn layering_mask(&mut self, mask: &BitVec, apply_color: RGB) -> Result<()>;
}

impl ApplyMask for Image {
    fn layering_mask(&mut self, mask: &BitVec, apply_color: RGB) -> Result<()> {
        let mut data = self.raw_data()?;
        let ptr = SafePtr::new(data.as_mut_ptr());

        if self.get_width() * self.get_height() != mask.len() as i32 {
            return Err(anyhow::anyhow!(
                "Mask size does not match image size, current image size: {}x{}, mask size: {}",
                self.get_width(),
                self.get_height(),
                mask.len()
            ));
        }

        (0..(self.get_width() * self.get_height() * 3) as usize)
            .into_par_iter()
            .for_each(|index| {
                if !unsafe { *mask.get_unchecked(index / 3) } {
                    return;
                }

                match (
                    index % 3,
                    apply_color.0 > 0,
                    apply_color.1 > 0,
                    apply_color.2 > 0,
                ) {
                    (0, true, _, _) => unsafe {
                        let deref = *ptr.add(index);
                        let after = deref.saturating_add(apply_color.0);
                        ptr.add(index).write(after);
                    },
                    (1, _, true, _) => unsafe {
                        let deref = *ptr.add(index);
                        let after = deref.saturating_add(apply_color.1);
                        ptr.add(index).write(after);
                    },
                    (2, _, _, true) => unsafe {
                        let deref = *ptr.add(index);
                        let after = deref.saturating_add(apply_color.2);
                        ptr.add(index).write(after);
                    },
                    _ => {}
                }
            });

        Ok(())
    }
}
