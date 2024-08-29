#![cfg_attr(debug_assertions, allow(warnings))]

fn main() {
    spark_inference::inference().unwrap();
    // spark_ffmpeg::get_pixels().unwrap();
}
