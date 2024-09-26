pub mod extractor;
pub mod masks;

pub(crate) struct SafeVecPtr<T>(*mut T);
impl<T> std::ops::Deref for SafeVecPtr<T> {
    type Target = *mut T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
unsafe impl<T> Send for SafeVecPtr<T> {}
unsafe impl<T> Sync for SafeVecPtr<T> {}