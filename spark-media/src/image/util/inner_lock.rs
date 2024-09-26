use parking_lot::RwLock;
use std::fmt::Debug;
use std::ops::Deref;

pub(crate) struct InnerLock<T>(RwLock<T>);

impl<T: Clone> Clone for InnerLock<T> {
    fn clone(&self) -> Self {
        let guard = self.0.write();
        InnerLock(RwLock::new(guard.clone()))
    }
}

impl<T: Debug> Debug for InnerLock<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let guard = self.0.read();
        guard.fmt(f)
    }
}

impl<T> Deref for InnerLock<T> {
    type Target = RwLock<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> InnerLock<T> {
    pub fn new(inner: T) -> Self {
        InnerLock(RwLock::new(inner))
    }

    pub fn into_inner(self) -> T {
        let guard = self.0.into_inner();
        guard
    }
}