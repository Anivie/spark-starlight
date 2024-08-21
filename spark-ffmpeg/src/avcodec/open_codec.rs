use crate::avcodec::AVCodec;
use crate::ffi::{avcodec_find_decoder, AVCodecID, AVCodecID_AV_CODEC_ID_PNG};
use anyhow::{anyhow, Result};
use std::ffi::CStr;

impl AVCodec {
    pub fn open_codec(id: AVCodecID) -> Result<Self> {
        let codec = unsafe {
            avcodec_find_decoder(id)
        };

        if codec.is_null() {
            Err(anyhow!("Failed to find codec"))
        } else {
            Ok(AVCodec { inner: codec.cast_mut() })
        }
    }
}

#[test]
fn test_open_codec() {
    let codec = AVCodec::open_codec(AVCodecID_AV_CODEC_ID_PNG).unwrap();
    let string = unsafe { CStr::from_ptr(codec.name) };
    assert_eq!(string.to_str().unwrap(), "png");
}