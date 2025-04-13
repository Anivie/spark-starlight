mod native;

use crate::inference::tts::native::{
    destroy_tts_engine, free_tts_audio, generate_tts_audio, init_tts_engine,
};
use std::ffi::CString;
use std::io::{BufWriter, Cursor, Write};
use std::str::FromStr;

pub fn tts() -> anyhow::Result<()> {
    unsafe {
        let tts = init_tts_engine(
            CString::from_str("./data/model/tts/matcha-zh/model-steps-3.onnx")?.as_ptr() as *const i8,
            CString::from_str("./data/model/tts/matcha-zh/hifigan_v3.onnx")?.as_ptr() as *const i8,
            CString::from_str("./data/model/tts/matcha-zh/lexicon.txt")?.as_ptr() as *const i8,
            CString::from_str("./data/model/tts/matcha-zh/tokens.txt")?.as_ptr() as *const i8,
            CString::from_str("./data/model/tts/matcha-zh/dict")?.as_ptr() as *const i8,
            CString::from_str("./data/model/tts/matcha-zh/date.fst,./data/model/tts/matcha-zh/number.fst,./data/model/tts/matcha-zh/phone.fst")?.as_ptr() as *const i8,
            16,
            0,
        );

        let back = generate_tts_audio(
            tts,
            CString::from_str("你好！世界上怎么会有这么可爱的猫")?.as_ptr() as *const i8,
            0,
            1.0,
        );

        let buf = {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: (*back).sample_rate as u32,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };

            let mut buf = Vec::new();
            let writer = BufWriter::new(Cursor::new(&mut buf));
            let mut writer = hound::WavWriter::new(writer, spec)?;
            // let mut writer = hound::WavWriter::create("sine.wav", spec).unwrap();
            for i in 0..(*back).n {
                let sample = *(*back).samples.offset(i as isize);
                writer.write_sample(sample).unwrap();
            }
            writer.finalize()?;
            free_tts_audio(back);
            buf
        };
        //write to file
        let mut file = std::fs::File::create("audio.wav")?;
        file.write_all(&buf)?;
        file.flush()?;
        destroy_tts_engine(tts);
    }
    println!("Created audio.wav");
    Ok(())
}
