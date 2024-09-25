use crate::avpacket::AVPacket;
use anyhow::Result;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::ptr::slice_from_raw_parts;

impl AVPacket {
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = File::create(path)?;
        let slice = unsafe {
            &*slice_from_raw_parts(self.data, self.size as usize)
        };
        file.write(slice)?;

        Ok(())
    }
}