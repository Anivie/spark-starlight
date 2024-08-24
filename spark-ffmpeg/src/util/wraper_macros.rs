macro_rules! wrap {
    (
        $(
            $name:ident
            $(<$($life:lifetime)*>)*
            $({$($field_name:ident : $field_value:ty, )*})*
            $(drop $drop:ident)*
            $(drop2 $drop2:ident)*
        ),+
    ) => {
        use std::ops::{Deref, DerefMut};

        paste::paste!{
            $(
                use crate::ffi::$name as [<$name Raw>];

                #[derive(Debug, Clone)]
                pub struct $name <$($($life,)*)*> {
                    pub(crate) inner: *mut [<$name Raw>],
                    $(
                    $(
                        $field_name: $field_value,
                    )*
                    )*
                }

                impl Deref for $name {
                    type Target = [<$name Raw>];

                    fn deref(&self) -> &Self::Target {
                        unsafe {
                            &*self.inner
                        }
                    }
                }

                impl DerefMut for $name {
                    fn deref_mut(&mut self) -> &mut Self::Target {
                        unsafe {
                            &mut *self.inner
                        }
                    }
                }

                $(
                    impl std::ops::Drop for $name {
                        fn drop(&mut self) {
                            unsafe {
                                crate::ffi::$drop(self.inner);
                            }
                        }
                    }
                )*

                $(
                    impl std::ops::Drop for $name {
                        fn drop(&mut self) {
                            unsafe {
                                crate::ffi::$drop2(&mut self.inner as *mut *mut [<$name Raw>]);
                            }
                        }
                    }
                )*
            )+
        }
    };
}