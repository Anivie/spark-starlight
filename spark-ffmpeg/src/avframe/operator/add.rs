use crate::avframe::AVFrame;
use anyhow::Result;
use ndarray::Array2;
use rayon::prelude::*;

impl AVFrame {
    pub fn layering_mask(&self, dim_index: usize, mask: &Array2<f32>) -> Result<()> {
        self
            .get_raw_data(dim_index)
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, value)| {
                if index % 3 == 1 {
                    let mask = mask[[index / 3 / 640, index / 3 % 640]];
                    if mask > 0. {
                        *value = if *value < 155 {
                            mask as u8 + *value
                        } else {
                            255
                        }
                    }
                }
            });

        Ok(())
    }
}