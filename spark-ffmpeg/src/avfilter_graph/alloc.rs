use crate::avfilter_context::AVFilterContext;
use crate::avfilter_graph::AVFilterGraph;
use crate::ffi::avfilter_graph_alloc;
use anyhow::{anyhow, Result};

impl AVFilterGraph {
    pub fn new() -> Result<Self> {
        let inner = unsafe {
            avfilter_graph_alloc()
        };

        if inner.is_null() {
            return Err(anyhow!("Could not allocate AVFilterGraph"));
        }

        Ok(Self {
            inner,
            contexts: vec![],
        })
    }

    pub fn add_context(&mut self, filter_name: &'static str, args: &'static str) -> Result<()> {
        let context = AVFilterContext::new(filter_name, args, self)?;
        self.contexts.push(context);

        Ok(())
    }
}

#[test]
fn test_context() {
    let mut context = AVFilterGraph::new().unwrap();
    context.add_context(
        "scale",
        "1920:1080:force_original_aspect_ratio=decrease"
    ).unwrap();

}