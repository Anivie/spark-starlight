use crate::ffi::AVStream;
use crate::ffi_enum::AVCodecID;

impl AVStream {
    pub fn codec_id(&self) -> AVCodecID {
        unsafe { AVCodecID::try_from((*self.codecpar).codec_id).unwrap() }
    }
}
