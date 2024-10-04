use std::ops::Deref;

pub struct SafePtr<T>(*mut T);
impl<T> SafePtr<T> {
    pub fn new(ptr: *mut T) -> Self {
        SafePtr(ptr)
    }
}

impl<T> Deref for SafePtr<T> {
    type Target = *mut T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
unsafe impl<T> Send for SafePtr<T> {}
unsafe impl<T> Sync for SafePtr<T> {}