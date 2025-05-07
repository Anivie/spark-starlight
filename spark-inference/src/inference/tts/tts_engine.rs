use crate::inference::tts::raw::{
    destroy_tts_engine, free_tts_audio, generate_tts_audio, init_tts_engine_en, init_tts_engine_zh,
};
use log::info;
use std::ffi::{c_void, CString};
use std::io::{BufWriter, Cursor};
use std::str::FromStr;

pub trait TTS {
    fn new_zh() -> anyhow::Result<Self>
    where
        Self: Sized;
    fn new_en() -> anyhow::Result<Self>
    where
        Self: Sized;
    fn generate<'a, T: Into<&'a str>>(&self, source: T) -> anyhow::Result<Vec<u8>>;
}

pub struct TTSEngine {
    engine: *const c_void,
}

unsafe impl Send for TTSEngine {}
unsafe impl Sync for TTSEngine {}

impl Drop for TTSEngine {
    fn drop(&mut self) {
        unsafe {
            destroy_tts_engine(self.engine);
        }
    }
}

impl TTS for TTSEngine {
    fn new_zh() -> anyhow::Result<Self> {
        let engine = unsafe {
            init_tts_engine_zh(
                CString::from_str("./data/model/tts/matcha-zh/model-steps-3.onnx")?.as_ptr() as *const i8,
                CString::from_str("./data/model/tts/matcha-zh/hifigan_v3.onnx")?.as_ptr() as *const i8,
                CString::from_str("./data/model/tts/matcha-zh/lexicon.txt")?.as_ptr() as *const i8,
                CString::from_str("./data/model/tts/matcha-zh/tokens.txt")?.as_ptr() as *const i8,
                CString::from_str("./data/model/tts/matcha-zh/dict")?.as_ptr() as *const i8,
                CString::from_str("./data/model/tts/matcha-zh/date.fst,./data/model/tts/matcha-zh/number.fst,./data/model/tts/matcha-zh/phone.fst")?.as_ptr() as *const i8,
                CString::from_str("cpu")?.as_ptr() as *const i8,
                16,
                0,
            )
        };
        if engine.is_null() {
            return Err(anyhow::anyhow!("Failed to initialize TTS engine"));
        }
        info!("TTS engine initialized successfully");
        Ok(Self { engine })
    }

    fn new_en() -> anyhow::Result<Self> {
        let engine = unsafe {
            init_tts_engine_en(
                CString::from_str("./data/model/tts/matcha-en/model-steps-3.onnx")?.as_ptr()
                    as *const i8,
                CString::from_str("./data/model/tts/matcha-en/vocos-22khz-univ.onnx")?.as_ptr()
                    as *const i8,
                CString::from_str("./data/model/tts/matcha-en/tokens.txt")?.as_ptr() as *const i8,
                CString::from_str("./data/model/tts/matcha-en/espeak-ng-data")?.as_ptr()
                    as *const i8,
                CString::from_str("cpu")?.as_ptr() as *const i8,
                16,
                0,
            )
        };
        if engine.is_null() {
            return Err(anyhow::anyhow!("Failed to initialize TTS engine"));
        }
        info!("TTS engine initialized successfully");
        Ok(Self { engine })
    }

    fn generate<'a, T: Into<&'a str>>(&self, source: T) -> anyhow::Result<Vec<u8>> {
        let source = source.into();
        if source.is_empty() {
            return Err(anyhow::anyhow!("Empty source string"));
        }

        let (back, sample_rate) = unsafe {
            let back = generate_tts_audio(
                self.engine,
                CString::from_str(source)?.as_ptr() as *const i8,
                0,
                1.0,
            );
            (back, (*back).sample_rate as u32)
        };

        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let mut buf = Vec::new();
        let writer = BufWriter::new(Cursor::new(&mut buf));
        let mut writer = hound::WavWriter::new(writer, spec)?;

        unsafe {
            for i in 0..(*back).n {
                let sample = *(*back).samples.offset(i as isize);
                writer.write_sample(sample)?;
            }
            writer.finalize()?;
            free_tts_audio(back);
        }

        Ok(buf)
    }
}
