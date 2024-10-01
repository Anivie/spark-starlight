use spark_proc_macro::wrap_ffmpeg;

pub mod new_packet;
pub mod file_io;

wrap_ffmpeg!(
    AVPacket drop+ [av_packet_free]
);