use crate::Image;
use anyhow::anyhow;
use std::ops::Deref;

impl Image {
    pub fn refresh_packet(&mut self) -> anyhow::Result<()> {
        let frame = self.inner.frame.as_ref().ok_or(anyhow!("Failed to get frame."))?;
        let frame = frame.read();

        let encoder = self.encoder.as_ref().ok_or(anyhow!("Encoder is not ready"))?;

        encoder.send_frame(&frame)?;
        drop(frame);

        let packet = encoder.receive_packet()?;
        self.inner.replace_packet(packet);

        Ok(())
    }

    pub fn refresh_frame(&mut self) -> anyhow::Result<()> {
        let packet = self.inner.packet.as_ref().ok_or(anyhow!("Failed to get packet."))?;
        let packet = packet.read();

        let decoder = self.decoder.as_ref().ok_or(anyhow!("Decoder is not ready"))?;

        decoder.send_packet(packet.deref())?;
        drop(packet);

        let frame = decoder.receive_frame()?;
        self.inner.replace_frame(frame);

        Ok(())
    }
}