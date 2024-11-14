use ndarray::prelude::*;

pub struct FormerState {
    object_memory: Array2<f32>,//Every obj_ptr for decoder output in each round
    mask_memory: Array4<f32>,//Every maskmem_features for memory encoder output in each round
    position_memory: Array4<f32>,//Every maskmem_features for decoder output in each round
}