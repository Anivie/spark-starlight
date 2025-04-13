use std::ffi::c_void;

extern "C" {
    pub fn init_tts_engine(
        acoustic_model: *const i8,
        vocoder: *const i8,
        lexicon: *const i8,
        tokens: *const i8,
        dict_dir: *const i8,  // Optional
        rule_fsts: *const i8, // Optional
        num_threads: i32,
        debug: i32,
    ) -> *const c_void;

    pub fn generate_tts_audio(
        engine_handle: *const c_void,
        text: *const i8,
        speaker_id: i32,
        speed: f32,
    ) -> *const SherpaOnnxGeneratedAudio;

    pub fn free_tts_audio(audio: *const SherpaOnnxGeneratedAudio);

    pub fn destroy_tts_engine(engine: *const c_void);
}

pub struct SherpaOnnxGeneratedAudio {
    pub samples: *const f32, // in the range [-1, 1]
    pub n: i32,              // number of samples
    pub sample_rate: i32,
}
