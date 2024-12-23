use crate::inference::sam::video_inference::inference_state::SamInferenceState;
use crate::utils::graph::SamPrompt;
use ort::session::SessionOutputs;

pub mod inference_state;
pub mod video_inference;

pub struct SamEncoderOutput<'a> {
    pub(super) encoder_output: SessionOutputs<'a, 'a>,
    pub(super) origin_size: (i32, i32),
}

pub enum InferenceInput {
    Prompt(SamPrompt<f32>),
    State(SamInferenceState),
}
