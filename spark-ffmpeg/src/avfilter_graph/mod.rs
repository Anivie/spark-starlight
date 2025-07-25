use crate::avfilter_context::AVFilterContext;
use spark_proc_macro::wrap_ffmpeg;

pub mod alloc;

wrap_ffmpeg!(
    AVFilterGraph {
        contexts: Vec<AVFilterContext>
    } drop+ [avfilter_graph_free]
);
