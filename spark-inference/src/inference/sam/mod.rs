use ort::value::DynValue;
use std::collections::HashMap;

pub struct SamEncoderOutput {
    pub(super) encoder_output: HashMap<String, DynValue>,
    pub(super) origin_size: (i32, i32),
}

pub mod image_inference;
