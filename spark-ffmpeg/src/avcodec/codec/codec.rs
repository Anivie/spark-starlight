use crate::avcodec::AVCodecContext;
use crate::avframe::AVFrame;
use crate::avpacket::AVPacket;
use crate::ffi::{avcodec_receive_frame, avcodec_receive_packet, avcodec_send_frame, avcodec_send_packet};
use anyhow::Result;

impl AVCodecContext {
    pub fn send_packet(&self, packet: &mut AVPacket) -> Result<()> {
        ffmpeg! {
            avcodec_send_packet(self.inner, packet.inner)
        }

        Ok(())
    }

    pub fn receive_packet(&self) -> Result<AVPacket> {
        let packet = AVPacket::new()?;
        ffmpeg! {
            avcodec_receive_packet(self.inner, packet.inner)
        }

        Ok(packet)
    }

    pub fn receive_frame(&self) -> Result<AVFrame> {
        let frame = AVFrame::new()?;
        ffmpeg! {
            avcodec_receive_frame(self.inner, frame.inner)
        }

        Ok(frame)
    }

    pub fn send_frame(&self, frame: &AVFrame) -> Result<()> {
        ffmpeg! {
            avcodec_send_frame(self.inner, frame.inner)
        }

        Ok(())
    }
}