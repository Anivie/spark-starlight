use std::ffi::c_void;

pub struct SherpaOnnxGeneratedAudio {
    pub samples: *const f32, // in the range [-1, 1]
    pub n: i32,              // number of samples
    pub sample_rate: i32,
}

extern "C" {
    // generate by gemini
    /**
     * @brief 初始化 TTS 引擎。
     *
     * @param acoustic_model 声学模型的路径。
     * @param vocoder 声码器模型的路径。
     * @param lexicon 词典文件的路径。
     * @param tokens tokens 文件的路径。
     * @param dict_dir (可选) Jieba 词典目录的路径，如果不需要可以为 NULL 或空字符串。
     * @param rule_fsts (可选) 规则 FSTs 的路径，多个路径用逗号分隔。如果不需要可以为 NULL 或空字符串。
     * @param num_threads 使用的线程数。
     * @param debug 是否启用调试日志 (1 表示启用, 0 表示禁用)。
     *
     * @return 成功则返回 TTS 引擎句柄 (void*)，失败则返回 NULL。
     *         使用完毕后，必须调用 destroy_tts_engine() 释放资源。
     */
    pub(super) fn init_tts_engine_zh(
        acoustic_model: *const i8,
        vocoder: *const i8,
        lexicon: *const i8,
        tokens: *const i8,
        dict_dir: *const i8,
        rule_fsts: *const i8,
        device: *const i8,
        num_threads: i32,
        debug: i32,
    ) -> *const c_void;
    pub(super) fn init_tts_engine_en(
        acoustic_model: *const i8,
        vocoder: *const i8,
        tokens: *const i8,
        data_dir: *const i8,
        device: *const i8,
        num_threads: i32,
        debug: i32,
    ) -> *const c_void;

    /**
     * @brief 使用 TTS 引擎生成音频。
     *
     * @param engine_handle 由 init_tts_engine() 返回的引擎句柄。
     * @param text 要转换为语音的文本。
     * @param speaker_id 说话人 ID。
     * @param speed 语速 (例如 1.0 代表正常语速, 更大则更快)。
     *
     * @return 成功则返回一个 const SherpaOnnxGeneratedAudio 对象，失败则返回 NULL。
     *         使用完毕后，必须调用 free_tts_audio() 释放此对象及其关联的音频数据。
     */
    pub(super) fn generate_tts_audio(
        engine_handle: *const c_void,
        text: *const i8,
        speaker_id: i32,
        speed: f32,
    ) -> *const SherpaOnnxGeneratedAudio;

    /**
     * @brief 释放由 generate_tts_audio() 创建的音频对象和相关资源。
     *
     * @param audio 指向 const SherpaOnnxGeneratedAudio 对象的指针。如果传入 NULL，则不执行任何操作。
     */
    pub(super) fn free_tts_audio(audio: *const SherpaOnnxGeneratedAudio);

    /**
     * @brief 销毁 TTS 引擎并释放所有相关资源。
     *
     * @param engine_handle 由 init_tts_engine() 返回的引擎句柄。如果传入 NULL，则不执行任何操作。
     */
    pub(super) fn destroy_tts_engine(engine: *const c_void);
}
