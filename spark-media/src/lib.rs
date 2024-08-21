use spark_ffmpeg::avformat::avformat_context::OpenFileToAVFormatContext;
use spark_ffmpeg::avformat::AVFormatContext;

#[test]
fn cat() {
    let mut format = AVFormatContext::open_file("./data/a.png", None).unwrap();
    format.video_stream().unwrap().for_each(|x| {

    });
}