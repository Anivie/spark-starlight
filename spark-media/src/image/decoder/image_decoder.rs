use crate::image::image::{ImageInner, ImageUtil};
use crate::image::util::inner_lock::InnerLock;
use crate::{Image, CODEC};
use anyhow::{bail, Result};
use log::warn;
use spark_ffmpeg::avcodec::{AVCodec, AVCodecContext};
use spark_ffmpeg::avformat::avformat_context::OpenFileToAVFormatContext;
use spark_ffmpeg::avformat::{AVFormatContext, AVMediaType};
use spark_ffmpeg::avframe::AVFrame;
use spark_ffmpeg::avpacket::AVPacket;

impl Image {
    pub fn open_file(path: impl Into<String>) -> Result<Self> {
        let mut format = AVFormatContext::open_file(path, None)?;
        let codec_context = format.video_stream()?.map(|(_, stream)| {
            let id = stream.codec_id();
            let codec = {
                let codec_guard = CODEC.read();
                codec_guard.get(&id).cloned()
            };
            match codec {
                Some(codec) => AVCodecContext::from_stream(&codec, stream, None),
                None => {
                    let codec = AVCodec::new_decoder_with_id(id)?;
                    let codec_context = AVCodecContext::from_stream(&codec, stream, None)?;
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

        let (packet, frame) = Self::decode(&codec_context, &mut format)?;

        let image = Image {
            decoder: Some(codec_context),
            encoder: None,
            inner: ImageInner {
                packet: Some(InnerLock::new(packet)),
                frame: Some(InnerLock::new(frame)),
            },
            utils: ImageUtil {
                sws: None,
                format: Some(format),
            },
        };

        Ok(image)
    }

    fn decode(codec: &AVCodecContext, format: &mut AVFormatContext) -> Result<(AVPacket, AVFrame)> {
        let mut vec: Vec<Result<(AVPacket, AVFrame)>> = format
            .frames(AVMediaType::VIDEO)?
            .map(|mut packet| {
                codec.send_packet(&mut packet)?;
                Ok((packet, codec.receive_frame()?))
            })
            .collect::<Vec<_>>();

        if vec.len() > 1 {
            warn!("More than one frame found, using the first one");
        }

        let packet = vec.remove(0)?;


        Ok((packet.0, packet.1.clone()))
    }
}