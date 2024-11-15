use crate::filter::filter::{AVFilter, Locked, UnLocked};
use anyhow::Result;
use spark_ffmpeg::avfilter_graph::AVFilterGraph;
use spark_ffmpeg::ffi_enum::AVPixelFormat;
use std::marker::PhantomData;
use spark_ffmpeg::avframe::AVFrame;

impl AVFilter {
    pub fn builder(pixel_format: AVPixelFormat, in_size: (i32, i32)) -> Result<AVFilter<UnLocked>> {
        let mut graph = AVFilterGraph::new()?;

        let arg = format!(
            "video_size={}x{}:pix_fmt={}:time_base={}/{}",
            in_size.0, in_size.1,
            pixel_format as i32,
            1, 30
        );
        graph.add_context_with_name("buffer", "in", Some(&arg))?;

        Ok(AVFilter {
            inner: graph,
            _marker: PhantomData,
        })
    }
}

impl AVFilter<Locked> {
    pub fn apply_image(&self, frame: &AVFrame) -> Result<AVFrame> {
        self.inner.apply_image(frame)
    }
}

impl AVFilter<UnLocked> {
    pub fn add_context(mut self, filter_name: &'static str, args: &'static str) -> Result<Self> {
        self.inner.add_context(filter_name, args)?;

        Ok(self)
    }

    pub fn build(mut self) -> Result<AVFilter<Locked>> {
        self.inner.add_context_with_name("buffersink", "out", None)?;
        self.inner.link()?;

        Ok(AVFilter {
            inner: self.inner,
            _marker: PhantomData,
        })
    }
}