use crate::image::util::image_inner::ImageInner;
use crate::image::util::image_util::ImageUtil;
use crate::{Image, CODEC};
use anyhow::{bail, Result};
use log::warn;
use spark_ffmpeg::av_io_context::AVIOContext;
use spark_ffmpeg::av_mem_alloc::AVMemorySegment;
use spark_ffmpeg::avcodec::{AVCodec, AVCodecContext};
use spark_ffmpeg::avformat::avformat_context::OpenFileToAVFormatContext;
use spark_ffmpeg::avformat::{AVFormatContext, AVMediaType};
use spark_ffmpeg::avframe::AVFrame;
use spark_ffmpeg::avpacket::AVPacket;
use std::mem::forget;
use std::path::Path;

impl Image {
    pub fn open_file(path: impl AsRef<Path>) -> Result<Self> {
        let mut format = AVFormatContext::open_file(path, None)?;
        let codec_context = format
            .video_stream()?
            .map(|(_, stream)| {
                let id = stream.codec_id();
                let codec_guard = CODEC.read();
                let codec = codec_guard.get(&id);
                match codec {
                    Some(codec) => AVCodecContext::from_stream(&codec, stream, None),
                    None => {
                        drop(codec_guard);
                        let codec = AVCodec::new_decoder_with_id(id)?;
                        let codec_context = AVCodecContext::from_stream(&codec, stream, None)?;
                        CODEC.write().insert(id, codec);
                        Ok(codec_context)
                    }
                }
            })
            .next();

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
                packet: Some(packet),
                frame,
            },
            utils: ImageUtil {
                sws: None,
                format: Some(format),
            },
        };

        Ok(image)
    }

    pub fn from_bytes<T: AsRef<[u8]>>(value: T) -> Result<Image> {
        let bytes = value.as_ref();
        let mut format = AVFormatContext::alloc()?;
        let memory_segment = AVMemorySegment::new(bytes.len())?;
        let mut io_context = AVIOContext::alloc(
            memory_segment.inner.cast(),
            bytes.len(),
            0,
            std::ptr::null_mut(),
            None,
            None,
            None,
        )?;
        io_context.fill_data(bytes)?;
        format.set_io_context(&io_context)?;

        forget(memory_segment);
        forget(io_context);

        let codec_context = format
            .video_stream()?
            .map(|(_, stream)| {
                let id = stream.codec_id();
                let codec_guard = CODEC.read();
                let codec = codec_guard.get(&id);
                match codec {
                    Some(codec) => AVCodecContext::from_stream(&codec, stream, None),
                    None => {
                        drop(codec_guard);
                        let codec = AVCodec::new_decoder_with_id(id)?;
                        let codec_context = AVCodecContext::from_stream(&codec, stream, None)?;
                        CODEC.write().insert(id, codec);
                        Ok(codec_context)
                    }
                }
            })
            .next();

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
                packet: Some(packet),
                frame,
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

        Ok((packet.0, packet.1))
    }
}
