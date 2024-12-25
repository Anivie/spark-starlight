#![feature(let_chains)]
#![cfg_attr(debug_assertions, allow(warnings))]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Result;
use spark_inference::disable_ffmpeg_logging;
use spark_inference::inference::sam::video_inference::inference_state::SamInferenceState;
use spark_inference::inference::sam::video_inference::video_inference::{
    SAMVideoInferenceSession, SamVideoInference,
};
use spark_inference::inference::sam::video_inference::InferenceInput;
use spark_inference::utils::graph::SamPrompt;
use spark_inference::utils::masks::ApplyMask;
use spark_media::filter::filter::AVFilter;
use spark_media::{Image, RGB};

fn main() -> Result<()> {
    disable_ffmpeg_logging();

    let sam2 = SAMVideoInferenceSession::new("./data/model/other3")?;
    let mut save_state: Option<SamInferenceState> = None;

    for index in 0..8 {
        let path = format!("./data/image/0000{index}.jpg");

        let image = Image::open_file(path.as_str())?;
        let encoded = sam2.encode_image(image)?;
        let (mask, state) = sam2.inference_frame(
            if save_state.is_some() {
                InferenceInput::State(save_state.take().unwrap())
            } else {
                InferenceInput::Prompt(SamPrompt::point(210., 350.))
            },
            &encoded,
        )?;

        let mut image = Image::open_file(path.as_str())?;
        let filter = AVFilter::builder(image.pixel_format()?, image.get_size())?
            .add_context("scale", "1024:1024")?
            .add_context("format", "rgb24")?
            .build()?;
        image.apply_filter(&filter)?;
        image.layering_mask(&mask, RGB(0, 65, 45))?;
        image.save_with_format(format!("./data/out/bird{index}_mask.png"))?;

        save_state.replace(state);
    }

    Ok(())
}
