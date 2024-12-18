use spark_ffmpeg::avfilter_graph::AVFilterGraph;
use std::marker::PhantomData;

pub struct AVFilter<T: FilterState = Locked> {
    pub(crate) inner: AVFilterGraph,
    pub(crate) _marker: PhantomData<T>,
}

pub struct Locked;
pub struct UnLocked;

pub trait FilterState {}

impl FilterState for Locked {}
impl FilterState for UnLocked {}
