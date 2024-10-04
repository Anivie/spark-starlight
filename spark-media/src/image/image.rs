use crate::image::util::image_inner::ImageInner;
use crate::image::util::image_util::ImageUtil;
use spark_ffmpeg::avcodec::AVCodecContext;
use spark_ffmpeg::DeepClone;

#[derive(Debug)]
pub struct Image {
    pub(super) inner: ImageInner,
    pub(super) utils: ImageUtil,

    pub(super) decoder: Option<AVCodecContext>,
    pub(super) encoder: Option<AVCodecContext>,
}

impl Image {
    pub fn tst_filter(&mut self) -> anyhow::Result<()> {
        use spark_ffmpeg::avfilter_graph::AVFilterGraph;

        let new_size = self.get_height().max(self.get_width());
        let pad_x = (new_size - self.get_width()) / 2;
        let pad_y = (new_size - self.get_height()) / 2;

        let mut graph = AVFilterGraph::new()?;
        graph.add_context("pad", "width=ih:height=ih:x=(ow-iw)/2:y=(oh-ih)/2:color=black")?;
        // graph.add_context("scale", "w=640:h=640:force_original_aspect_ratio=decrease")?;
        // graph.add_context("pad", "width=640:height=640:x=(ow-iw)/2:y=(oh-ih)/2:color=black")?;
        let frame = graph.apply_image(&self.inner.frame)?;
        self.inner.frame = frame;

        Ok(())
    }
}

impl Clone for Image {
    fn clone(&self) -> Self {
        Self {
            utils: Default::default(),
            inner: ImageInner {
                packet: None,
                frame: self.inner.frame.deep_clone().unwrap(),
            },
            decoder: self.decoder.as_ref().map(|x| x.deep_clone()).transpose().unwrap(),
            encoder: self.encoder.as_ref().map(|x| x.deep_clone()).transpose().unwrap(),
        }
    }
}