use std::sync::atomic::AtomicBool;
use std::sync::LazyLock;

static IS_INIT: LazyLock<AtomicBool> = LazyLock::new(|| AtomicBool::new(false));

pub mod inference_engine;
pub mod run;
mod entity;