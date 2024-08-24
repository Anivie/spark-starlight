use crate::av_mem_alloc::AVMemorySegment;
use crate::avframe::AVFrame;
use crate::pixel::pixel_formater::PixelIterator;

pub struct AVFrameCollector<'a> {
    frames: Vec<&'a AVFrame>,
    _segment: Option<AVMemorySegment>,
}

impl<'a> AVFrameCollector<'a> {
    pub fn new(frames: Vec<&'a AVFrame>, _segment: Option<AVMemorySegment>) -> Self {
        AVFrameCollector {
            frames,
            _segment,
        }
    }

    pub fn frames(&self, frame_index: usize) -> PixelIterator {
        PixelIterator {
            current_x: 0,
            current_y: 0,
            frame: self.frames[frame_index],
        }
    }
}