use crate::Image;
use anyhow::Result;

impl Image {
    pub fn replace_raw_data(&mut self, data: &[u8]) -> Result<()> {
        self.inner.frame.replace_raw_data(data);

        Ok(())
    }

    pub fn fill_data(&mut self, data: &[u8]) -> Result<()> {
        let pixel_format = self.available_codec().pixel_format();
        let frame = &mut self.inner.frame;
        frame.fill_data(data, pixel_format);

        Ok(())
    }
}