use crate::avpacket::AVPacket;
use crate::ffi::av_packet_alloc;
use anyhow::{bail, Result};

impl AVPacket {
    pub fn new() -> Result<Self> {
        let packet = unsafe {
            av_packet_alloc()
        };

        if packet.is_null() {
            bail!("Failed to allocate memory by 'av_packet_alloc'.");
        }

        Ok(AVPacket { inner: packet })
    }
}