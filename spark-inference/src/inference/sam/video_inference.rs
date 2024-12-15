use crate::engine::inference_engine::{ExecutionProvider, OnnxSession};
use crate::inference::sam::former_state::{FormerState, InferenceResult, InferenceType};
use crate::inference::sam::image_inference::SAM2ImageInferenceSession;
use anyhow::Result;
use bitvec::prelude::*;
use ndarray::{Array2, Array3, Array4, ArrayViewD};
use ort::inputs;
use ort::session::SessionOutputs;
use spark_media::filter::filter::AVFilter;
use spark_media::Image;
use std::ops::Deref;
use std::path::Path;

pub trait SamVideoInference {
    fn inference_sam(&self, types: InferenceType, image: &mut Image) -> Result<InferenceResult>;
}

pub struct SAM2VideoInferenceSession {
    image_session: SAM2ImageInferenceSession,
    memory_attention: OnnxSession,
    memory_encoder: OnnxSession,
}

impl SAM2VideoInferenceSession {
    pub fn new(folder_path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            image_session: SAM2ImageInferenceSession::raw(
                OnnxSession::new(folder_path.as_ref().join("image_encoder.onnx"), ExecutionProvider::CUDA)?,
                OnnxSession::new(folder_path.as_ref().join("image_decoder.onnx"), ExecutionProvider::CUDA)?,
            ),
            memory_attention: OnnxSession::new(folder_path.as_ref().join("memory_attention.onnx"), ExecutionProvider::CUDA)?,
            memory_encoder: OnnxSession::new(folder_path.as_ref().join("memory_encoder.onnx"), ExecutionProvider::CUDA)?,
        })
    }
}

impl SamVideoInference for SAM2VideoInferenceSession {
    fn inference_sam(&self, types: InferenceType, image: &mut Image) -> Result<InferenceResult> {
        let (image, image_size) = {
            let filter = AVFilter::builder(image.pixel_format()?, image.get_size())?
                .add_context("scale", "1024:1024")?
                .add_context("format", "rgb24")?
                .build()?;

            let image_size = image.get_size();
            image.apply_filter(&filter)?;
            (image, image_size)
        };

        let encoder_output = self.image_session.inference_image_encoder(image.raw_data()?.deref())?;

        let mut memory_attention_output = None;
        let vision_feats = {
            match &types {
                InferenceType::First(_) => encoder_output["vision_feats"].try_extract_tensor::<f32>()?,
                InferenceType::WithState(state) => {
                    let memory_attention_output_inner = self.inference_memory_attention(
                        encoder_output["vision_feats"].try_extract_tensor::<f32>()?.view(),
                        encoder_output["vision_pos_embed"].try_extract_tensor::<f32>()?.view(),
                        state.object_memory(),
                        state.mask_memory(),
                        &state.memory_pos_embed()?,
                    )?;

                    memory_attention_output.replace(memory_attention_output_inner);
                    let outputs = memory_attention_output.as_ref().unwrap();
                    outputs["image_embed"].try_extract_tensor::<f32>()?
                }
            }
        };

        let decoder_output = self.image_session.inference_image_decoder(
            image_size,
            vision_feats,
            encoder_output["high_res_feat0"].try_extract_tensor::<f32>()?,
            encoder_output["high_res_feat1"].try_extract_tensor::<f32>()?,
            types.get_inner(),
        )?;

        let memory_encoder_output = self.inference_memory_encoder(
            decoder_output["mask_for_mem"].try_extract_tensor::<f32>()?.view(),
            encoder_output["pix_feat"].try_extract_tensor::<f32>()?.view()
        )?;

        let state = match types {
            InferenceType::First(point) => {
                FormerState::new(&decoder_output, &memory_encoder_output, point)?
            }
            InferenceType::WithState(state) => {
                state.update(&decoder_output, &memory_encoder_output)?
            }
        };

        let mask = {
            let pred_mask = decoder_output["pred_mask"].try_extract_tensor::<f32>()?;
            let mut back = BitVec::with_capacity(pred_mask.len());
            pred_mask.iter().for_each(|x| {
                back.push(*x > 0f32);
            });
            back
        };

        Ok(InferenceResult {
            mask,
            state,
        })
    }
}

impl SAM2VideoInferenceSession {
    fn inference_memory_attention(
        &self,
        vision_feats: ArrayViewD<f32>, //Every vision_feats for encoder output in each round
        vision_pos_embed: ArrayViewD<f32>, //Every vision_pos_embed for encoder output in each round

        object_memory: &Array2<f32>, //Every obj_ptr for decoder output in each round
        mask_memory: &Array4<f32>, //Every maskmem_features for memory encoder output in each round
        mask_position_embedded: &Array3<f32>, //Every maskmem_pos_enc + temporal_code for decoder output in each round
    ) -> Result<SessionOutputs> {
        let result = self.memory_attention.run(inputs![
            "current_vision_feat" => vision_feats,
            "current_vision_pos_embed" => vision_pos_embed,
            "memory_0" => object_memory.clone(),
            "memory_1" => mask_memory.clone(),
            "memory_pos_embed" => mask_position_embedded.clone(),
        ]?)?;

        Ok(result)
    }

    fn inference_memory_encoder(
        &self,
        mask_for_mem: ArrayViewD<f32>,
        pix_feat: ArrayViewD<f32>,
    ) -> Result<SessionOutputs> {
        let result = self.memory_encoder.run(inputs![
            "mask_for_mem" => mask_for_mem,
            "pix_feat" => pix_feat,
        ]?)?;

        Ok(result)
    }
}