use crate::ffi::AVStream;

pub type AVCodecID = u32;

impl AVStream {
    pub fn codec_id(&self) -> AVCodecID {
        unsafe {
            (*self.codecpar).codec_id
        }
    }
}