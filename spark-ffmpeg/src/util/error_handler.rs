use anyhow::anyhow;
use std::ffi::c_int;

#[allow(dead_code)]
pub(crate) trait ToError {
    fn to_error(self, message: &str) -> anyhow::Result<()>;
}

impl ToError for c_int {
    fn to_error(self, message: &str) -> anyhow::Result<()> {
        Err(anyhow!("Error when calling native function: {}, error code: {}", message, self))
    }
}