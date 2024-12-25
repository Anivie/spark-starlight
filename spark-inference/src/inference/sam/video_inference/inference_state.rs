use crate::inference::sam::video_inference::video_inference::SAMVideoInferenceSession;
use crate::inference::sam::video_inference::SamEncoderOutput;
use crate::utils::graph::SamPrompt;
use crate::utils::tensor::sigmoid;
use anyhow::Result;
use bitvec::vec::BitVec;
use ndarray::concatenate;
use ndarray::prelude::*;
use ndarray_npy::ReadNpyExt;
use ort::inputs;
use ort::session::SessionOutputs;
use ort::value::{Tensor, TensorRef};
use std::fs::File;

pub enum InferenceType {
    First(SamPrompt<f32>),
    WithState(SamInferenceState),
}

pub struct InferenceResult {
    pub mask: BitVec,
    pub state: SamInferenceState,
}

impl InferenceType {
    pub(crate) fn get_inner(&self) -> &SamPrompt<f32> {
        match self {
            InferenceType::First(points) => points,
            InferenceType::WithState(state) => &state.prompt,
        }
    }
}

pub struct SamInferenceState {
    pub(super) pix_feat: Array4<f32>,

    pub(super) memory1: Array3<f32>,
    pub(super) memory2: Array3<f32>,

    pub(super) memory_pos1: Array3<f32>,
    pub(super) memory_pos2: Array3<f32>,

    pub(super) prompt: SamPrompt<f32>,
    pub(super) max_len: usize,
}

impl SamInferenceState {
    pub(crate) fn update(
        &mut self,
        instance: &SAMVideoInferenceSession,
        encoded_result: &SamEncoderOutput,
        mask_decoder_output: &SessionOutputs,
        pred_mask: Array2<f32>,
    ) -> Result<()> {
        let curr = encoded_result.encoder_output["backbone_fpn_2"].try_extract_tensor::<f32>()?;
        let curr = curr.into_shape_with_order((1, 256, 4096))?;
        let curr = curr.permuted_axes([2, 0, 1]);

        let curr_pos =
            encoded_result.encoder_output["vision_pos_enc_2"].try_extract_tensor::<f32>()?;
        let curr_pos = curr_pos.into_shape_with_order((1, 256, 4096))?;
        let curr_pos = curr_pos.permuted_axes([2, 0, 1]);

        let masks = sigmoid(pred_mask);
        let masks = masks.to_shape((1, 1, 1024, 1024))?;

        let mem_encode_result = {
            let pix_feat =
                encoded_result.encoder_output["backbone_fpn_2"].try_extract_tensor::<f32>()?;
            let mem_encode_result = instance.memory_encoder.run(inputs![
                "pix_feat" => TensorRef::from_array_view(pix_feat)?,
                "masks" => TensorRef::from_array_view(masks.view())?,
            ])?;
            mem_encode_result
        };

        let (vision_features, vision_pos_enc) = (
            mem_encode_result["vision_features"].try_extract_tensor::<f32>()?,
            mem_encode_result["vision_pos_enc"].try_extract_tensor::<f32>()?,
        );

        let memory1 = {
            let memory = vision_features.into_shape_with_order((1, 64, 4096))?;
            let memory1 = memory.permuted_axes([2, 0, 1]);
            if memory1.shape()[0] > self.max_len * 4096 {
                let memory1 = self.memory1.slice(s![4096.., .., ..]).to_owned();
                concatenate![Axis(0), self.memory1, memory1]
            } else {
                concatenate![Axis(0), self.memory1, memory1]
            }
        };

        let obj_ptr = {
            let sam_out = mask_decoder_output["sam_tokens_out"].try_extract_tensor::<f32>()?;
            let sam_out = sam_out.into_shape_with_order((4, 256))?;
            let sam_out = sam_out.slice(s![1, ..]);
            let sam_out = sam_out.into_shape_with_order((1, 256))?;
            instance
                .mlp
                .run(inputs![TensorRef::from_array_view(sam_out)?])?
        };
        let obj_ptr = obj_ptr["x_out"].try_extract_tensor::<f32>()?;
        let memory2 = obj_ptr.view();
        let memory2 = memory2.into_shape_with_order((1, 4, 64))?;
        let memory2 = memory2.permuted_axes([1, 0, 2]);
        let mut memory2 = if memory2.shape()[0] > self.max_len * 4 {
            let memory2 = self.memory2.slice(s![4.., .., ..]).to_owned();
            concatenate![Axis(0), self.memory2, memory2]
        } else {
            concatenate![Axis(0), self.memory2, memory2]
        };

        let memory_pos = vision_pos_enc.into_shape_with_order((1, 64, 4096))?;
        let memory_pos1 = memory_pos.permuted_axes([2, 0, 1]);
        let mut memory_pos1 = if memory_pos1.shape()[0] > self.max_len * 4096 {
            let memory_pos1 = self.memory_pos1.slice(s![4096.., .., ..]).to_owned();
            concatenate![Axis(0), self.memory_pos1, memory_pos1]
        } else {
            concatenate![Axis(0), self.memory_pos1, memory_pos1]
        };

        let obj_pos = get_1d_sine_pe(
            Array1::range(0.0, (self.memory_pos2.shape()[0] / 4) as f32 / 4.0, 1.0),
            256,
            10000.0,
        )?;
        let memory_pos2 = {
            let memory_pos2 = {
                instance
                    .obj_ptr_tpos_proj
                    .run(inputs![Tensor::from_array(obj_pos)?])?
            };
            let memory_pos2 = memory_pos2["x_out"].try_extract_tensor::<f32>()?;
            let dim0 = memory_pos2.shape()[0];
            let memory_pos2 = memory_pos2.into_shape_with_order((dim0, 1, 64))?;
            let mut new_memory_pos2 = memory_pos2.to_owned();
            if dim0 < 4 {
                for _ in 0..(4 - dim0) {
                    new_memory_pos2 = concatenate![
                        Axis(0),
                        new_memory_pos2,
                        memory_pos2.slice(s![-1.., .., ..])
                    ];
                }
            }

            new_memory_pos2
        };
        let mut memory_pos2 = if memory_pos2.shape()[0] > self.max_len * 4 {
            let memory_pos2 = self.memory_pos2.slice(s![4.., .., ..]).to_owned();
            concatenate![Axis(0), self.memory_pos2, memory_pos2]
        } else {
            concatenate![Axis(0), self.memory_pos2, memory_pos2]
        };

        let attention_mask1 = Array::<bool, _>::from_shape_vec(
            (memory1.shape()[0], memory1.shape()[1]),
            vec![true; memory1.shape()[0] * memory1.shape()[1]],
        )?;
        let attention_mask2 = Array::<bool, _>::from_shape_vec(
            (memory2.shape()[0], memory2.shape()[1]),
            vec![true; memory2.shape()[0] * memory2.shape()[1]],
        )?;

        let attention_results = instance.memory_attention.run(inputs![
            "curr" => Tensor::from_array(curr.to_owned())?,
            "memory_1" => Tensor::from_array(memory1.clone())?,
            "memory_2" => TensorRef::from_array_view(memory2.view())?,
            "curr_pos" => Tensor::from_array(curr_pos.to_owned())?,
            "memory_pos_1" => Tensor::from_array(memory_pos1.clone())?,
            "memory_pos_2" => TensorRef::from_array_view(memory_pos2.view())?,
            "attention_mask_1" => Tensor::from_array(attention_mask1)?,
            "attention_mask_2" => Tensor::from_array(attention_mask2)?,
        ])?;

        let pix_feat = attention_results["pix_feat"].try_extract_tensor::<f32>()?;
        let pix_feat = pix_feat.into_shape_with_order((4096, 1, 256))?;
        let pix_feat = pix_feat.permuted_axes([1, 2, 0]);
        let pix_feat = pix_feat.to_shape((1, 256, 64, 64))?;

        self.pix_feat = pix_feat.to_shape((1, 256, 64, 64))?.to_owned();
        self.memory1 = memory1.to_owned();
        self.memory2 = memory2.to_owned();
        self.memory_pos1 = memory_pos1.to_owned();
        self.memory_pos2 = memory_pos2;

        Ok(())
    }

    pub(crate) fn new(
        instance: &SAMVideoInferenceSession,
        encoded_result: &SamEncoderOutput,
        mask_decoder_output: &SessionOutputs,
        pred_mask: Array2<f32>,
        prompt: SamPrompt<f32>,
    ) -> Result<Self> {
        let curr = Array4::<f32>::read_npy(File::open("./data/other/b2.npy")?)?;
        // let curr = encoded_result.encoder_output["backbone_fpn_2"].try_extract_tensor::<f32>()?;
        let curr = curr.into_shape_with_order((1, 256, 4096))?;
        let curr = curr.permuted_axes([2, 0, 1]);

        let curr_pos =
            encoded_result.encoder_output["vision_pos_enc_2"].try_extract_tensor::<f32>()?;
        let curr_pos = curr_pos.into_shape_with_order((1, 256, 4096))?;
        let curr_pos = curr_pos.permuted_axes([2, 0, 1]);

        let masks = sigmoid(pred_mask);
        let masks = masks.to_shape((1, 1, 1024, 1024))?;

        let mem_encode_result = {
            let pix_feat =
                encoded_result.encoder_output["backbone_fpn_2"].try_extract_tensor::<f32>()?;
            let mem_encode_result = instance.memory_encoder.run(inputs![
                "pix_feat" => TensorRef::from_array_view(pix_feat)?,
                "masks" => TensorRef::from_array_view(masks.view())?,
            ])?;
            mem_encode_result
        };

        let (vision_features, vision_pos_enc) = (
            mem_encode_result["vision_features"].try_extract_tensor::<f32>()?,
            mem_encode_result["vision_pos_enc"].try_extract_tensor::<f32>()?,
        );

        let memory1 = {
            let memory = vision_features.into_shape_with_order((1, 64, 4096))?;
            let memory1 = memory.permuted_axes([2, 0, 1]);
            memory1.to_owned()
        };

        let obj_ptr = {
            let sam_out = mask_decoder_output["sam_tokens_out"].try_extract_tensor::<f32>()?;
            let sam_out = sam_out.into_shape_with_order((4, 256))?;
            let sam_out = sam_out.slice(s![1, ..]);
            let sam_out = sam_out.into_shape_with_order((1, 256))?;
            instance
                .mlp
                .run(inputs![TensorRef::from_array_view(sam_out)?])?
        };
        let obj_ptr = obj_ptr["x_out"].try_extract_tensor::<f32>()?;
        let memory2 = obj_ptr.view();
        let memory2 = memory2.into_shape_with_order((1, 4, 64))?;
        let memory2 = memory2.permuted_axes([1, 0, 2]);

        let memory_pos = vision_pos_enc.into_shape_with_order((1, 64, 4096))?;
        let memory_pos1 = memory_pos.permuted_axes([2, 0, 1]);
        let memory_pos1 = memory_pos1.to_owned();

        let obj_pos = get_1d_sine_pe(array![1.], 256, 10000.0)?;
        let memory_pos2 = {
            let memory_pos2 = {
                let back = obj_pos.view();
                instance
                    .obj_ptr_tpos_proj
                    .run(inputs![TensorRef::from_array_view(back)?])?
            };
            let memory_pos2 = memory_pos2["x_out"].try_extract_tensor::<f32>()?;
            let memory_pos2 = memory_pos2.into_shape_with_order((1, 1, 64))?;
            let new_memory_pos2 = concatenate![Axis(0), memory_pos2, memory_pos2];
            let new_memory_pos2 = concatenate![Axis(0), new_memory_pos2, memory_pos2];
            let new_memory_pos2 = concatenate![Axis(0), new_memory_pos2, memory_pos2];
            new_memory_pos2
        };

        let attention_mask1 = Array::<bool, _>::from_shape_vec(
            (memory1.shape()[0], memory1.shape()[1]),
            vec![true; memory1.shape()[0] * memory1.shape()[1]],
        )?;
        let attention_mask2 = Array::<bool, _>::from_shape_vec(
            (memory2.shape()[0], memory2.shape()[1]),
            vec![true; memory2.shape()[0] * memory2.shape()[1]],
        )?;

        // let curr = Array3::<f32>::read_npy(File::open("./data/other/curr.npy")?)?;
        // let curr_pos = Array3::<f32>::read_npy(File::open("./data/other/curr_pos.npy")?)?;
        // let memory1 = Array3::<f32>::read_npy(File::open("./data/other/memory1.npy")?)?;
        // let memory2 = Array3::<f32>::read_npy(File::open("./data/other/memory2.npy")?)?;
        // let memory_pos1 = Array3::<f32>::read_npy(File::open("./data/other/mp1.npy")?)?;
        // let memory_pos2 = Array3::<f32>::read_npy(File::open("./data/other/mp2.npy")?)?;
        // let attention_mask1 = Array2::<bool>::read_npy(File::open("./data/other/a1.npy")?)?;
        // let attention_mask2 = Array2::<bool>::read_npy(File::open("./data/other/a2.npy")?)?;
        let attention_results = instance.memory_attention.run(inputs![
            "curr" => Tensor::from_array(curr.to_owned())?,
            "memory_1" => Tensor::from_array(memory1.clone())?,
            "memory_2" => TensorRef::from_array_view(memory2.view())?,
            "curr_pos" => Tensor::from_array(curr_pos.to_owned())?,
            "memory_pos_1" => Tensor::from_array(memory_pos1.clone())?,
            "memory_pos_2" => TensorRef::from_array_view(memory_pos2.view())?,
            "attention_mask_1" => Tensor::from_array(attention_mask1)?,
            "attention_mask_2" => Tensor::from_array(attention_mask2)?,
        ])?;

        let pix_feat = attention_results["pix_feat"].try_extract_tensor::<f32>()?;
        let pix_feat = pix_feat.into_shape_with_order((4096, 1, 256))?;
        let pix_feat = pix_feat.permuted_axes([1, 2, 0]);
        let pix_feat = pix_feat.to_shape((1, 256, 64, 64))?;

        Ok(SamInferenceState {
            pix_feat: pix_feat.to_owned(),
            memory1,
            memory2: memory2.to_owned(),
            memory_pos1,
            memory_pos2: memory_pos2.to_owned(),
            prompt,
            max_len: 8,
        })
    }
}

fn get_1d_sine_pe(pos_inds: Array1<f32>, dim: usize, temperature: f32) -> Result<Array2<f32>> {
    let pe_dim = dim / 2;
    let mut dim_t: Array1<f32> = Array1::range(0.0, pe_dim as f32, 1.0);
    dim_t.mapv_inplace(|x| temperature.powf(2.0 * (x / 2.0).floor() / pe_dim as f32));

    let pos_inds_expanded = pos_inds.insert_axis(Axis(1)); // 对应于Python的unsqueeze(-1)
    let mut pos_embed = pos_inds_expanded / &dim_t;

    let sin_pos_embed = pos_embed.mapv(|x| x.sin());
    let cos_pos_embed = pos_embed.mapv(|x| x.cos());
    pos_embed = ndarray::concatenate(Axis(1), &[sin_pos_embed.view(), cos_pos_embed.view()])?;

    Ok(pos_embed)
}
