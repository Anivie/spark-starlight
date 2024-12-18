use crate::ffi::{av_free, av_malloc};
use anyhow::{bail, Result};
use std::ffi::c_void;
use std::ops::Deref;

#[derive(Debug, Clone)]
pub struct AVMemorySegment {
    pub inner: *mut c_void,
    pub size: usize,
}

impl Deref for AVMemorySegment {
    type Target = *mut c_void;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl Drop for AVMemorySegment {
    fn drop(&mut self) {
        unsafe {
            av_free(self.inner);
        }
    }
}

impl AVMemorySegment {
    pub fn new(size: usize) -> Result<Self> {
        let segment = unsafe { av_malloc(size) };

        if segment.is_null() {
            bail!("Failed to allocate memory by 'av_malloc'.");
        }

        Ok(AVMemorySegment {
            inner: segment,
            size,
        })
    }
}
