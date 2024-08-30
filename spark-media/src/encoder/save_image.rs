use crate::image_util::extract::ExtraToTensor;
use crate::Image;
use anyhow::Result;
use spark_ffmpeg::avcodec::{AVCodec, AVCodecContext};
use spark_ffmpeg::avpacket::AVPacket;
use spark_ffmpeg::avstream::AVCodecID;
use spark_ffmpeg::pixformat::AVPixelFormat;

impl Image {
    pub fn from_data(size: (i32, i32), pixel_format: AVPixelFormat, codec_id: AVCodecID) -> Result<Self> {
        let codec = AVCodec::new_encoder_with_id(codec_id)?;

        let mut codec_context = AVCodecContext::new_save(&codec, size, pixel_format, 400000)?;

        Ok(Image {
            format: None,
            codec: codec_context,
        })
    }

    pub fn fill_data(&mut self, data: &[u8]) -> Result<AVPacket> {
        self.codec.fill_data(&data);
        self.codec.send_frame(None)?;
        let mut av_packet = AVPacket::new()?;
        self.codec.receive_packet(&mut av_packet)?;

        Ok(av_packet)
    }
}

#[test]
fn test_encoder_and_decoder() {
    let nao: fn() -> Result<()> = || {
        use rayon::prelude::*;

        println!("Start reading image");
        let mut input = Image::open("/home/spark-starlight/data/image/b.png")?;
        println!("Finish reading image");
        input.decode()?;
        println!("Finish decode image");
        let frame = input.resize((640, 640), AVPixelFormat::AvPixFmtRgb24)?;
        let tensor = frame.extra_standard_image_to_tensor()?;
        let tensor: Vec<u8> = tensor
            .par_iter()
            .map(|&x| (255. * x) as u8)
            .collect();
        println!("Finish converting image to tensor");
        // tensor[0..100].iter().for_each(|x| print!("{}, ", x));

        let mut image = Image::from_data((640, 640), AVPixelFormat::AvPixFmtGray8, 61)?;
        let packet = image.fill_data(tensor.as_slice())?;
        packet.save("/home/spark-starlight/data/out/test_ead.png")?;

        let mut image = Image::from_data((641, 641), AVPixelFormat::AvPixFmtRgb24, 61)?;
        let frame = input.resize((641, 641), AVPixelFormat::AvPixFmtRgb24)?;
        let packet = image.fill_data(frame.get_raw_data(0).as_slice())?;
        packet.save("/home/spark-starlight/data/out/test_ead_r.png")?;

        Ok(())
    };

    nao().unwrap();
}