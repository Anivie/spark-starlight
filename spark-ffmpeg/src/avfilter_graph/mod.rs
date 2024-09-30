mod alloc;

wrap!(
    AVFilterGraph drop2 avfilter_graph_free
);