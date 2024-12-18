use crate::avformat::{AVFormatContext, AVMediaType};
use crate::avpacket::AVPacket;
use anyhow::{anyhow, Result};

pub struct AVFormatContextFrames<'a> {
    pub(super) context: &'a AVFormatContext,
    pub(super) video_index: &'a Vec<u32>,
}

impl Iterator for AVFormatContextFrames<'_> {
    type Item = AVPacket;

    fn next(&mut self) -> Option<Self::Item> {
        let mut packet = AVPacket::new().ok()?;

        while (packet.stream_index as u32) <= *self.video_index.last()? {
            self.context.read_frame(&mut packet).ok()?;
            if self.video_index.contains(&(packet.stream_index as u32)) {
                return Some(packet);
            }
        }

        None
    }
}

impl AVFormatContext {
    pub fn frames(&mut self, target_type: AVMediaType) -> Result<AVFormatContextFrames> {
        let vec = self
            .scanned_stream
            .get(&target_type)
            .ok_or(anyhow!("No target stream {:?} found", target_type))?;

        let packet = AVFormatContextFrames {
            context: self,
            video_index: vec,
        };

        Ok(packet)
    }
}
