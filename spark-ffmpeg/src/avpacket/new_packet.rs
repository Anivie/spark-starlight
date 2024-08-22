use crate::avpacket::AVPacket;
use crate::ffi::{av_packet_alloc, av_packet_unref};
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

    pub fn release(&mut self) {
        unsafe {
            av_packet_unref(self.inner);
        }
    }
}