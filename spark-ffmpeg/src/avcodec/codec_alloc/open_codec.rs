use crate::avcodec::AVCodec;
use crate::ffi::{avcodec_find_decoder, avcodec_find_encoder, AVCodecID, AVStream};
use anyhow::{anyhow, Result};

impl AVCodec {
    pub fn new_decoder(stream: &AVStream) -> Result<Self> {
        let codec_id = unsafe { (*stream.codecpar).codec_id };
        Self::new_decoder_with_id(codec_id)
    }
    
    pub fn new_decoder_with_id(id: AVCodecID) -> Result<Self> {
        let codec = unsafe {
            avcodec_find_decoder(id)
        };

        if codec.is_null() {
            Err(anyhow!("Failed to find codec"))
        } else {
            Ok(AVCodec { inner: codec.cast_mut() })
        }
    }
    pub fn new_encoder(stream: &AVStream) -> Result<Self> {
        let codec_id = unsafe { (*stream.codecpar).codec_id };
        Self::new_decoder_with_id(codec_id)
    }

    pub fn new_encoder_with_id(id: AVCodecID) -> Result<Self> {
        let codec = unsafe {
            avcodec_find_encoder(id)
        };

        if codec.is_null() {
            Err(anyhow!("Failed to find codec"))
        } else {
            Ok(AVCodec { inner: codec.cast_mut() })
        }
    }


}