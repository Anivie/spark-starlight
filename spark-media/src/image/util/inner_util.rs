use crate::Image;
use anyhow::anyhow;

impl Image {
    pub fn refresh_packet(&mut self) -> anyhow::Result<()> {
        let encoder = self.encoder.as_ref().ok_or(anyhow!("Encoder is not ready"))?;

        encoder.send_frame(&self.inner.frame)?;

        let packet = encoder.receive_packet()?;
        self.inner.packet = Some(packet);

        Ok(())
    }

    pub fn refresh_frame(&mut self) -> anyhow::Result<()> {
        let packet = self.inner.packet.as_ref().ok_or(anyhow!("Failed to get packet."))?;
        let decoder = self.decoder.as_ref().ok_or(anyhow!("Decoder is not ready"))?;

        decoder.send_packet(packet)?;

        let frame = decoder.receive_frame()?;
        self.inner.frame = frame;

        Ok(())
    }
}