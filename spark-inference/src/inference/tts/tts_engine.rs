use crate::inference::tts::raw::{
    destroy_tts_engine, free_tts_audio, generate_tts_audio, init_tts_engine,
};
use std::ffi::{c_void, CString};
use std::io::{BufWriter, Cursor};
use std::str::FromStr;

pub trait TTS {
    fn new() -> anyhow::Result<Self>
    where
        Self: Sized;
    fn generate(&self, source: &str) -> anyhow::Result<Vec<u8>>;
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

impl TTSEngine {
    fn new_inner(
        acoustic_model: &str,
        vocoder: &str,
        lexicon: &str,
        tokens: &str,
        dict_dir: &str,
        rule_fsts: &str,
        device: &str,
        num_threads: i32,
        debug: i32,
    ) -> anyhow::Result<Self> {
        let engine = unsafe {
            init_tts_engine(
                CString::from_str(acoustic_model)?.as_ptr() as *const i8,
                CString::from_str(vocoder)?.as_ptr() as *const i8,
                CString::from_str(lexicon)?.as_ptr() as *const i8,
                CString::from_str(tokens)?.as_ptr() as *const i8,
                CString::from_str(dict_dir)?.as_ptr() as *const i8,
                CString::from_str(rule_fsts)?.as_ptr() as *const i8,
                CString::from_str(device)?.as_ptr() as *const i8,
                num_threads,
                debug,
            )
        };
        Ok(TTSEngine { engine })
    }
}

impl TTS for TTSEngine {
    fn new() -> anyhow::Result<Self> {
        Ok(Self::new_inner(
            "./data/model/tts/matcha-zh/model-steps-3.onnx",
            "./data/model/tts/matcha-zh/hifigan_v3.onnx",
            "./data/model/tts/matcha-zh/lexicon.txt",
            "./data/model/tts/matcha-zh/tokens.txt",
            "./data/model/tts/matcha-zh/dict",
            "./data/model/tts/matcha-zh/date.fst,./data/model/tts/matcha-zh/number.fst,./data/model/tts/matcha-zh/phone.fst",
            "cuda",
            16,
            0,
        )?)
    }

    fn generate(&self, source: &str) -> anyhow::Result<Vec<u8>> {
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
