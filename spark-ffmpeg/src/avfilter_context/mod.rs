use spark_proc_macro::wrap_ffmpeg;

mod alloc;
mod apply;

wrap_ffmpeg!(
    AVFilterContext/* drop [avfilter_free]*/
);