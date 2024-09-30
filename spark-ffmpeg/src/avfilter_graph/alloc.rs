use crate::avfilter_graph::AVFilterGraph;
use crate::ffi::avfilter_graph_alloc;
use anyhow::{anyhow, Result};

impl AVFilterGraph {
    pub(crate) fn new() -> Result<Self> {
        let ptr = unsafe {
            avfilter_graph_alloc()
        };

        if ptr.is_null() {
            return Err(anyhow!("Could not allocate AVFilterGraph"));
        }

        Ok(Self {
            inner: ptr,
        })
    }
}