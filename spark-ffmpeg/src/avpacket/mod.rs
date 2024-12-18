use spark_proc_macro::wrap_ffmpeg;

pub mod file_io;
pub mod new_packet;

wrap_ffmpeg!(
    AVPacket drop+ [av_packet_free]
);
