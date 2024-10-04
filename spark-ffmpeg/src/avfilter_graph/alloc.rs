use std::ptr::null_mut;
use crate::avfilter_context::AVFilterContext;
use crate::avfilter_graph::AVFilterGraph;
use crate::ffi::{av_buffersink_get_frame, avfilter_graph_alloc, avfilter_graph_config, avfilter_link, av_buffersrc_add_frame};
use anyhow::{anyhow, Result};
use crate::avframe::AVFrame;

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
            linked: false,
        })
    }

    pub fn apply_image(&mut self, image: &AVFrame) -> Result<AVFrame> {
        self.link()?;

        ffmpeg! {
            av_buffersrc_add_frame(self.contexts[0].inner, image.inner)
        }

        let back = AVFrame::new()?;
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

#[test]
fn test_context() {
    let mut context = AVFilterGraph::new().unwrap();
    context.add_context(
        "scale",
        "1920:1080:force_original_aspect_ratio=decrease"
    ).unwrap();

}