use crate::avfilter_context::{AVFilterContext, AVFilterContextRaw};
use crate::avfilter_graph::AVFilterGraph;
use crate::ffi::{avfilter_free, avfilter_get_by_name, avfilter_graph_create_filter, avfilter_graph_free};
use anyhow::{bail, Result};
use std::ffi::{CStr, CString};
use std::ptr::null_mut;

impl AVFilterContext {
    pub fn new(filter_name: &'static str, args: &'static str) -> Result<Self> {
        let graph = AVFilterGraph::new()?;

        let filter_name = CString::new(filter_name)?;
        let args = CString::new(args)?;

        let filter = unsafe {
            avfilter_get_by_name(filter_name.clone().into_raw())
        };

        if filter.is_null() {
            bail!("Could not find filter with name: {}", filter_name.to_string_lossy());
        }

        let mut inner = null_mut::<AVFilterContextRaw>();
        ffmpeg! {
            avfilter_graph_create_filter(
                &mut inner,
                filter,
                filter_name.into_raw(),
                args.into_raw(),
                null_mut(),
                graph.inner,
            )
        }

        Ok(Self {
            inner,
            graph,
        })
    }
}

#[test]
fn test_context() {
    let context = AVFilterContext::new(
        "scale",
        "1920:1080:force_original_aspect_ratio=decrease"
    ).unwrap();
    assert!(!context.inner.is_null());
}