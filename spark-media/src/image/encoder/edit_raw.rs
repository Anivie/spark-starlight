use crate::Image;
use anyhow::Result;

impl Image {
    pub fn replace_raw_data(&mut self, data: &[u8]) -> Result<()> {
        self.inner.frame.replace_raw_data(data);

        Ok(())
    }

    pub fn fill_data(&mut self, data: &[u8]) -> Result<()> {
        &self.inner.frame.fill_data(data, self.available_codec().pixel_format());
        Ok(())
    }
}