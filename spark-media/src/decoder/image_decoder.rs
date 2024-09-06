use crate::{Image, CODEC};
use anyhow::{bail, Result};
use log::warn;
use spark_ffmpeg::avcodec::{AVCodec, AVCodecContext};
use spark_ffmpeg::avformat::avformat_context::OpenFileToAVFormatContext;
use spark_ffmpeg::avformat::{AVFormatContext, AVMediaType};
use spark_ffmpeg::avframe::AVFrame;
use spark_ffmpeg::pixformat::AVPixelFormat;
use spark_ffmpeg::sws::SwsContext;

impl Image {
    pub fn open(path: impl Into<String>) -> Result<Self> {
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

        Ok(Image {
            format: Some(format),
            codec: codec_context,
            sws: None,
        })
    }

    pub fn decode(&mut self) -> Result<&AVFrame> {
        if self.format.is_none() {
            bail!("Missing format context to decode");
        }
        let format = self.format.as_mut().unwrap();

        let mut vec = format
            .frames(AVMediaType::VIDEO)?
            .map(|mut packet| {
                self.codec.send_packet(&mut packet)?;
                self.codec.receive_frame()
            })
            .collect::<Vec<_>>();
        if vec.len() > 1 {
            warn!("More than one frame found, using the first one");
        }

        let frame = vec.remove(0)?;

        Ok(frame)
    }

    pub fn resize(&mut self, size: (i32, i32), format: AVPixelFormat) -> Result<AVFrame> {
        let sws = match &self.sws {
            Some(sws) => sws,
            None => {
                let sws = SwsContext::from_format_context(&self.codec, Some(format), Some((size.0, size.1)), None)?;
                self.sws = Some(sws);
                self.sws.as_ref().unwrap()
            }
        };

        let scaled_frame = {
            let mut scaled_frame = AVFrame::new()?;
            scaled_frame.set_size(size.0, size.1);
            scaled_frame.alloc_image(format, size.0, size.1)?;
            scaled_frame
        };

        sws.scale_image(self.codec.last_frame(), &scaled_frame)?;

        Ok(scaled_frame)
    }
}