use crate::Image;
use anyhow::{anyhow, Result};

impl Image {
    pub fn replace_data(&mut self, data: &[u8]) -> Result<()> {
        let frame = self.inner.frame
            .as_ref()
            .ok_or(anyhow!("No frame found"))?;

        frame.write().replace_raw_data(data);

        Ok(())
    }

    pub fn fill_data(&mut self, data: &[u8]) -> Result<()> {
        // self.available_codec().send_frame(None)?;

        let av_packet = self.available_codec().receive_packet()?;
        self.inner.replace_packet(av_packet);

        Ok(())
    }
}