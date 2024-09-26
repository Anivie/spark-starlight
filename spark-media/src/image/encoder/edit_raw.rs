use crate::Image;
use anyhow::{anyhow, Result};

impl Image {
    pub fn replace_raw_data(&mut self, data: &[u8]) -> Result<()> {
        let frame = self.inner.frame
            .as_ref()
            .ok_or(anyhow!("No frame found"))?;

        frame.write().replace_raw_data(data);

        Ok(())
    }

    pub fn fill_data(&mut self, data: &[u8]) -> Result<()> {
        let mut guard = self.inner.frame.as_ref().ok_or(anyhow!("No frame found"))?.write();
        guard.fill_data(data, self.available_codec().pixel_format());
        Ok(())
    }
}