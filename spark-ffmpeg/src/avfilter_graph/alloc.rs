use crate::avfilter_context::AVFilterContext;
use crate::avfilter_graph::AVFilterGraph;
use crate::avframe::AVFrame;
use crate::ffi::{av_buffersink_get_frame, av_buffersrc_add_frame, avfilter_graph_alloc, avfilter_graph_config, avfilter_link};
use crate::CloneFrom;
use anyhow::{anyhow, Result};
use std::ptr::null_mut;

impl AVFilterGraph {
    pub fn new() -> Result<Self> {
        let inner = unsafe {
            avfilter_graph_alloc()
        };

        if inner.is_null() {
            return Err(anyhow!("Could not allocate AVFilterGraph"));
        }

        Ok(Self {
            inner,
            contexts: vec![],
        })
    }

    pub fn apply_image(&self, image: &AVFrame) -> Result<AVFrame> {
        ffmpeg! {
            av_buffersrc_add_frame(self.contexts[0].inner, image.inner)
        }

        let mut back = AVFrame::new()?;
        back.clone_fields_from(image);
        ffmpeg! {
            av_buffersink_get_frame(self.contexts[self.contexts.len() - 1].inner, back.inner)
        }

        Ok(back)
    }

    pub fn add_context(&mut self, filter_name: &str, args: &str) -> Result<()> {
        let context = AVFilterContext::new(filter_name, args, self)?;
        self.contexts.push(context);

        Ok(())
    }

    pub fn add_context_with_name(&mut self, name: &'static str, filter_name: &'static str, args: Option<&str>) -> Result<()> {
        let context = AVFilterContext::new_with(name, filter_name, args, self)?;
        self.contexts.push(context);

        Ok(())
    }

    pub fn link(&mut self) -> Result<()> {
        if self.contexts.len() > 1 {
            for index in 0..self.contexts.len() - 1 {
                let src = self.contexts[index].inner;
                let dst = self.contexts[index + 1].inner;

                ffmpeg! {
                    avfilter_link(src, 0, dst, 0)
                }
            }

            ffmpeg! {
                avfilter_graph_config(self.inner, null_mut())
            }
        }

        Ok(())
    }
}