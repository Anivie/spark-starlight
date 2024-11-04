use crate::filter::filter::{AVFilter, Locked};
use crate::Image;

impl Image {
    pub fn apply_filter(&mut self, filter: &AVFilter<Locked>) -> anyhow::Result<()> {
        let frame = filter.apply_image(&self.inner.frame)?;
        self.inner.frame = frame;

        Ok(())
    }
}