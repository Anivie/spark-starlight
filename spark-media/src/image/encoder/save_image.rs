use std::path::Path;
use crate::Image;
use anyhow::Result;
use spark_ffmpeg::avcodec::{AVCodec, AVCodecContext};
use spark_ffmpeg::avstream::AVCodecID;
use spark_ffmpeg::pixformat::AVPixelFormat;

impl Image {
    pub fn new_with_empty(size: (i32, i32), pixel_format: AVPixelFormat, codec_id: AVCodecID) -> Result<Self> {
        let codec = AVCodec::new_encoder_with_id(codec_id)?;

        let codec_context = AVCodecContext::new_save(
            &codec, size, pixel_format, 400000
        )?;

        Ok(Image {
            packet: None,
            decoder: None,
            encoder: Some(codec_context),
            utils: Default::default(),
        })
    }

    pub fn replace_data(&mut self, data: &[u8]) -> Result<()> {
        self.codec.replace_data(data);

        Ok(())
    }

    pub fn fill_data(&mut self, data: &[u8]) -> Result<()> {
        self.codec.fill_data(&data);
        self.codec.send_frame(None)?;

        let av_packet = self.codec.receive_packet()?;
        self.packet = Some(av_packet);

        Ok(())
    }

    pub fn save(&mut self, path: impl AsRef<Path>) -> Result<()> {
        match &self.packet {
            Some(packet) => packet.save(path)?,
            None => {
                self.codec.send_frame(None)?;

                let mut packet = self.codec.receive_packet()?;
                packet.save(path)?;

                self.packet = Some(packet);
            }
        }

        Ok(())
    }
}

#[test]
fn test_encoder_and_decoder() {
    use crate::image::util::extract::ExtraToTensor;
    use crate::image::decoder::size::ResizeImage;
    let nao: fn() -> Result<()> = || {
        use rayon::prelude::*;

        println!("Start reading image");
        let mut input = Image::open_file("/home/spark-starlight/data/image/b.png")?;
        println!("Finish reading image");
        input.resize_to((640, 640))?;
        let tensor = input.extra_standard_image_to_tensor()?;
        let tensor: Vec<u8> = tensor
            .par_iter()
            .map(|&x| (255. * x) as u8)
            .collect();
        println!("Finish converting image to tensor");
        // tensor[0..100].iter().for_each(|x| print!("{}, ", x));

        let mut image = Image::new_with_empty((640, 640), AVPixelFormat::AvPixFmtGray8, 61)?;
        image.fill_data(tensor.as_slice())?;
        image.save("/home/spark-starlight/data/out/test_ead.png")?;

        let mut image = Image::new_with_empty((641, 641), AVPixelFormat::AvPixFmtRgb24, 61)?;
        input.resize_to((641, 641))?;
        // let packet = image.fill_data(frame.get_raw_data(0).as_slice())?;
        image.save("/home/spark-starlight/data/out/test_ead_r.png")?;

        Ok(())
    };

    nao().unwrap();
}