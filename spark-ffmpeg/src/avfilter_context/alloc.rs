use crate::avfilter_context::{AVFilterContext, AVFilterContextRaw};
use crate::avfilter_graph::AVFilterGraph;
use crate::ffi::{avfilter_get_by_name, avfilter_graph_create_filter};
use anyhow::{bail, Result};
use std::ffi::CString;
use std::ptr::null_mut;

impl AVFilterContext {
    pub(crate) fn new(filter_name: &str, args: &str, graph: &AVFilterGraph) -> Result<Self> {
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
            inner
        })
    }

    pub(crate) fn new_with(filter: &str, filter_name: &str, args: Option<&str>, graph: &AVFilterGraph) -> Result<Self> {
        let filter = CString::new(filter)?;
        let filter_name = CString::new(filter_name)?;
        let args = args.map(|x| CString::new(x)).transpose()?;

        let filter = unsafe {
            avfilter_get_by_name(filter.into_raw())
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
                args.map(|x| x.into_raw()).unwrap_or(null_mut()),
                null_mut(),
                graph.inner,
            )
        }

        Ok(Self {
            inner
        })
    }
}