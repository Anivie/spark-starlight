use crate::avfilter_graph::AVFilterGraph;

mod alloc;

wrap!(
    AVFilterContext {
        graph: AVFilterGraph,
    }drop avfilter_free
);