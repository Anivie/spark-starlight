use anyhow::{bail, Result};
use bitvec::vec::BitVec;
use ndarray::concatenate;
use ndarray::prelude::*;
use ort::session::SessionOutputs;
use crate::utils::graph::Point;

pub enum InferenceType {
    First(Vec<Point<u32>>),
    WithState(FormerState)
}

pub struct InferenceResult {
    pub mask: BitVec,
    pub state: FormerState
}

impl InferenceType {
    pub(crate) fn get_points(&self) -> &Vec<Point<u32>> {
        match self {
            InferenceType::First(points) => points,
            InferenceType::WithState(state) => &state.points
        }
    }
}

pub struct FormerState {
    object_memory: Array2<f32>,//Every obj_ptr for decoder output in each round
    mask_memory: Array4<f32>,//Every maskmem_features for memory encoder output in each round
    mask_mem_pos_enc: Vec<Array3<f32>>,//Every maskmem_features for decoder output in each round
    temporal_code: Array3<f32>,//Every maskmem_features for decoder output in each round

    points: Vec<Point<u32>>,
}

impl FormerState {
    pub fn object_memory(&self) -> &Array2<f32> {
        &self.object_memory
    }

    pub fn mask_memory(&self) -> &Array4<f32> {
        &self.mask_memory
    }

    pub fn points(&self) -> &Vec<Point<u32>> {
        &self.points
    }

    pub fn memory_pos_embed(&self) -> Result<Array3<f32>> {
        if self.mask_mem_pos_enc.is_empty() {
            bail!("'memory_pos_embed' should not be called when mask_mem_pos_enc is empty!")
        }

        let mut back = &self.mask_mem_pos_enc[0] + &self.temporal_code.slice(s![0, .., ..]);
        back = concatenate![Axis(0), Array3::zeros((4, 1, 64)), back];

        for index in 1..self.mask_mem_pos_enc.len() {
            let mut tmp = &self.mask_mem_pos_enc[index] + &self.temporal_code.slice(s![index, .., ..]);
            tmp = concatenate![Axis(0), Array3::zeros((4, 1, 64)), tmp];
            back = concatenate![Axis(0), back, tmp];
        }

        Ok(back)
    }
}

impl FormerState {
    pub(crate) fn new(
        image_decoder_output: &SessionOutputs,
        memory_encoder_output: &SessionOutputs,
        points: Vec<Point<u32>>
    ) -> Result<Self> {
        let object_memory = image_decoder_output["obj_ptr"].try_extract_tensor::<f32>()?.to_owned();
        let object_memory = object_memory.into_dimensionality::<Ix2>()?;

        let mask_memory = memory_encoder_output["maskmem_features"].try_extract_tensor::<f32>()?.to_owned();
        let mask_memory = mask_memory.into_dimensionality::<Ix4>()?;

        let mask_mem_pos_enc = memory_encoder_output["maskmem_pos_enc"].try_extract_tensor::<f32>()?;
        let mask_mem_pos_enc = mask_mem_pos_enc.to_shape((4096, 1, 64))?;
        let mask_mem_pos_enc = vec![mask_mem_pos_enc.to_owned()];

        let temporal_code = memory_encoder_output["temporal_code"].try_extract_tensor::<f32>()?;
        let temporal_code = temporal_code.to_shape((7, 1, 64))?.to_owned();

        Ok(FormerState {
            object_memory,
            mask_memory,
            mask_mem_pos_enc,
            temporal_code,
            points
        })
    }

    pub(crate) fn update(
        self,
        image_decoder_output: &SessionOutputs,
        memory_encoder_output: &SessionOutputs,
    ) -> Result<Self> {
        let object_memory = {
            let object_memory = image_decoder_output["obj_ptr"].try_extract_tensor::<f32>()?.to_owned();
            let object_memory = object_memory.into_dimensionality::<Ix2>()?;

            let last_object_memory = if self.object_memory.shape()[0] > 16 {
                self.object_memory.slice(s![1.., ..]).to_owned()
            } else {
                self.object_memory
            };

            concatenate![Axis(0), last_object_memory, object_memory]
        };

        let mask_memory = {
            let mask_memory = memory_encoder_output["maskmem_features"].try_extract_tensor::<f32>()?.to_owned();
            let mask_memory = mask_memory.into_dimensionality::<Ix4>()?;

            let last_mask_memory = if self.mask_memory.shape()[0] > 7 {
                self.mask_memory.slice(s![1.., .., .., ..]).to_owned()
            } else {
                self.mask_memory
            };

            concatenate![Axis(0), last_mask_memory, mask_memory]
        };

        let mask_mem_pos_enc = {
            let position_encoded = memory_encoder_output["maskmem_pos_enc"].try_extract_tensor::<f32>()?;
            let position_encoded = position_encoded.to_shape((4096, 1, 64))?.to_owned();
            let mut mask_mem_pos_enc = self.mask_mem_pos_enc;

            if mask_mem_pos_enc.len() > 7 {
                mask_mem_pos_enc.remove(0);
            }

            mask_mem_pos_enc.push(position_encoded);
            mask_mem_pos_enc
        };

        let temporal_code = memory_encoder_output["temporal_code"].try_extract_tensor::<f32>()?;
        let temporal_code = temporal_code.to_shape((7, 1, 64))?.to_owned();

        Ok(FormerState {
            object_memory,
            mask_memory,
            mask_mem_pos_enc,
            temporal_code,
            points: self.points
        })
    }
}