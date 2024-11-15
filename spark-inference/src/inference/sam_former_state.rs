use anyhow::Result;
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

impl InferenceResult {
    pub(crate) fn new(mask: BitVec, state: FormerState) -> Self {
        InferenceResult {
            mask,
            state
        }
    }
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
    mask_pos_embed: Array3<f32>,//Every maskmem_features for decoder output in each round

    points: Vec<Point<u32>>,
}

impl FormerState {
    pub fn object_memory(&self) -> &Array2<f32> {
        &self.object_memory
    }

    pub fn mask_memory(&self) -> &Array4<f32> {
        &self.mask_memory
    }

    pub fn mask_pos_embed(&self) -> &Array3<f32> {
        &self.mask_pos_embed
    }

    pub fn points(&self) -> &Vec<Point<u32>> {
        &self.points
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
        let mask_memory = mask_memory.into_shape_with_order((1, 64, 64, 64))?;

        let mask_pos_embed = {
            let position_encoded = memory_encoder_output["maskmem_pos_enc"].try_extract_tensor::<f32>()?;
            let position_encoded = position_encoded.to_shape((4096, 1, 64))?;

            let time_code = memory_encoder_output["temporal_code"].try_extract_tensor::<f32>()?;
            let time_code = time_code.to_shape((7, 1, 64))?;
            let time_code = concatenate![Axis(0), time_code, Array3::zeros((4096 - 7, 1, 64))];
            let mask_pos_embed = position_encoded + time_code;
            mask_pos_embed.into_dimensionality()?.to_owned()
        };

        Ok(FormerState {
            object_memory,
            mask_memory,
            mask_pos_embed,
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

            let last_mask_memory = if self.mask_memory.shape()[0] > 16 {
                self.mask_memory.slice(s![1.., .., .., ..]).to_owned()
            } else {
                self.mask_memory
            };

            concatenate![Axis(0), last_mask_memory, mask_memory]
        };

        let mask_pos_embed = {
            let position_encoded = memory_encoder_output["maskmem_pos_enc"].try_extract_tensor::<f32>()?;
            let position_encoded = position_encoded.to_shape((4096, 1, 64))?;

            let time_code = memory_encoder_output["temporal_code"].try_extract_tensor::<f32>()?;
            let time_code = time_code.to_shape((7, 1, 64))?;
            let time_code = concatenate![Axis(0), time_code, Array3::zeros((4096 - 7, 1, 64))];
            let memory_pos_embed = position_encoded + time_code;
            let memory_pos_embed = memory_pos_embed.into_dimensionality::<Ix3>()?.to_owned();

            /*let last_mask_pos_embed = if self.mask_pos_embed.shape()[0] > 16 {
                self.mask_pos_embed.slice(s![1.., .., ..]).to_owned()
            } else {
                self.mask_pos_embed
            };*/

            let last_mask_pos_embed =self.mask_pos_embed;
            concatenate![Axis(0), last_mask_pos_embed, memory_pos_embed]
        };

        Ok(FormerState {
            object_memory,
            mask_memory,
            mask_pos_embed,
            points: self.points
        })
    }
}