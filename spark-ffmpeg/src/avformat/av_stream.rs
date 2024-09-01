use crate::avformat::{AVFormatContext, AVMediaType};
use crate::ffi::{avformat_find_stream_info, AVDictionary, AVStream};
use anyhow::anyhow;
use std::ptr::null_mut;

pub struct AVFormatContextStream<'a> {
    pub(super) context: &'a AVFormatContext,
    pub(super) video_index: Vec<u32>,
    pub(super) current_index: usize,
}

impl<'a> Iterator for AVFormatContextStream<'a> {
    type Item = (u32, &'a AVStream);

    fn next(&mut self) -> Option<Self::Item> {
        let index = *self.video_index.get(self.current_index)?;
        let stream = unsafe {
            &**self.context.streams.wrapping_add(index as usize)
        };
        self.current_index += 1;

        Some((index, stream))
    }
}

impl AVFormatContext {
    pub fn find_stream(&mut self, target_type: AVMediaType) -> anyhow::Result<Vec<u32>> {
        if let Some(ok) = self.scanned_stream.get(&target_type) {
            return Ok(ok.clone());
        }

        if !self.opened {
            ffmpeg! {
                avformat_find_stream_info(
                    self.inner,
                    null_mut::<*mut AVDictionary>()
                ) or "Failed to find stream"
            }
            self.opened = true;
        }

        let first_match_stream = (0..self.nb_streams)
            .into_iter()
            .filter(|x| {
                unsafe {
                    (*(**self.streams.offset(*x as isize)).codecpar).codec_type == target_type as i32
                }
            })
            .collect::<Vec<_>>();

        if !first_match_stream.is_empty() {
            self.scanned_stream.insert(target_type, first_match_stream.clone());
            Ok(first_match_stream)
        }else {
            Err(anyhow!("No target stream {:?} found", target_type))
        }
    }

    pub fn stream(&mut self, target_type: AVMediaType) -> anyhow::Result<AVFormatContextStream> {
        let vec = self.find_stream(target_type)?;

        let stream = AVFormatContextStream {
            context: self,
            video_index: vec,
            current_index: 0,
        };

        Ok(stream)
    }

    pub fn video_stream(&mut self) -> anyhow::Result<AVFormatContextStream> {
        Ok(self.stream(AVMediaType::VIDEO)?)
    }

    pub fn audio_stream(&mut self) -> anyhow::Result<AVFormatContextStream> {
        Ok(self.stream(AVMediaType::AUDIO)?)
    }
}