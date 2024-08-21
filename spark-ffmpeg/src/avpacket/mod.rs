mod new_packet;

wrap!(
    AVPacket drop2 av_packet_free
);