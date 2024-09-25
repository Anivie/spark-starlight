use crate::image::image::ImageUtil;
use crate::{Image, CODEC};
use anyhow::{bail, Result};
use log::warn;
use spark_ffmpeg::avcodec::{AVCodec, AVCodecContext};
use spark_ffmpeg::avformat::avformat_context::OpenFileToAVFormatContext;
use spark_ffmpeg::avformat::{AVFormatContext, AVMediaType};
use spark_ffmpeg::avpacket::AVPacket;

impl Image {
    pub fn open_file(path: impl Into<String>) -> Result<Self> {
        let mut format = AVFormatContext::open_file(path, None)?;
        let codec_context = format.video_stream()?.map(|(_, stream)| {
            let id = stream.codec_id();
            let codec_guard = CODEC.read();
            let codec = codec_guard.get(&id).cloned();
            match codec {
                Some(codec) => AVCodecContext::new(&codec, stream, None),
                None => {
                    drop(codec_guard);
                    let codec = AVCodec::new_decoder_with_id(id)?;
                    let codec_context = AVCodecContext::new(&codec, stream, None)?;
                    CODEC.write().insert(id, codec);
                    Ok(codec_context)
                }
            }
        }).next();

        let codec_context = match codec_context {
            Some(Ok(codec_context)) => codec_context,
            Some(Err(e)) => bail!("Failed to open codec: {}", e),
            None => bail!("No video stream found"),
        };

        let packet = Self::decode(&codec_context, &mut format)?;

        let image = Image {
            decoder: Some(codec_context),
            packet: Some(packet),
            encoder: None,
            utils: ImageUtil {
                sws: None,
                format: Some(format),
            },
        };

        Ok(image)
    }

    fn decode(codec: &AVCodecContext, format: &mut AVFormatContext) -> Result<AVPacket> {
        let mut vec = format
            .frames(AVMediaType::VIDEO)?
            .map(|mut packet| {
                codec.send_packet(&mut packet)?;
                codec.receive_frame()?;
                Ok(packet)
            })
            .collect::<Vec<_>>();

        if vec.len() > 1 {
            warn!("More than one frame found, using the first one");
        }

        let packet: Result<AVPacket> = vec.remove(0);

        Ok(packet?)
    }
}