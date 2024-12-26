use ort::session::SessionOutputs;

pub struct SamEncoderOutput<'a> {
    pub(super) encoder_output: SessionOutputs<'a, 'a>,
    pub(super) origin_size: (i32, i32),
}

pub mod image_inference;
