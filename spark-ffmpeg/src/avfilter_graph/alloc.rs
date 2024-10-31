use crate::avfilter_context::AVFilterContext;
use crate::avfilter_graph::AVFilterGraph;
use crate::avframe::AVFrame;
use crate::ffi::{av_buffersink_get_frame, av_buffersrc_add_frame, avfilter_graph_alloc, avfilter_graph_config, avfilter_link};
use crate::pixformat::AVPixelFormat;
use crate::CloneFrom;
use anyhow::{anyhow, Result};
use std::ptr::null_mut;

impl AVFilterGraph {
    pub fn new(pixel_format: AVPixelFormat, in_size: (i32, i32)) -> Result<Self> {
        let inner = unsafe {
            avfilter_graph_alloc()
        };

        if inner.is_null() {
            return Err(anyhow!("Could not allocate AVFilterGraph"));
        }

        let mut graph = Self {
            inner,
            contexts: vec![],
            linked: false,
            locked: false,
        };

        let arg = format!(
            "video_size={}x{}:pix_fmt={}:time_base={}/{}",
            in_size.0, in_size.1,
            pixel_format as i32,
            1, 30
        );
        let context = AVFilterContext::new_with("buffer", "in", Some(&arg), &graph)?;
        graph.contexts.push(context);

        Ok(graph)
    }

    pub fn apply_image(&mut self, image: &AVFrame) -> Result<AVFrame> {
        if !self.locked {
            let context = AVFilterContext::new_with("buffersink", "out", None, self)?;
            self.contexts.push(context);
            self.locked = true;
        }

        self.link()?;

        ffmpeg! {
            av_buffersrc_add_frame(self.contexts[0].inner, image.inner)
        }

        let mut back = AVFrame::new()?;
        back.clone_copy_fields(image);
        ffmpeg! {
            av_buffersink_get_frame(self.contexts[self.contexts.len() - 1].inner, back.inner)
        }

        Ok(back)
    }

    pub fn add_context(&mut self, filter_name: &'static str, args: &'static str) -> Result<()> {
        let context = AVFilterContext::new(filter_name, args, self)?;
        self.contexts.push(context);

        if self.linked {
            self.linked = false;
        }

        Ok(())
    }

    fn link(&mut self) -> Result<()> {
        if self.linked { return Ok(()); }

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

        self.linked = true;

        Ok(())
    }
}