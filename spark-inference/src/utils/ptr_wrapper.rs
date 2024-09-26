use std::ops::Deref;

pub(crate) struct SafeVecPtr<T>(*mut T);
impl<T> SafeVecPtr<T> {
    pub(crate) fn new(ptr: *mut T) -> Self {
        SafeVecPtr(ptr)
    }
}

impl<T> Deref for SafeVecPtr<T> {
    type Target = *mut T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
unsafe impl<T> Send for SafeVecPtr<T> {}
unsafe impl<T> Sync for SafeVecPtr<T> {}