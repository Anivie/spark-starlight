use spark_proc_macro::wrap_ffmpeg;

mod new_sws;

wrap_ffmpeg!(
  SwsContext drop [sws_freeContext]
);