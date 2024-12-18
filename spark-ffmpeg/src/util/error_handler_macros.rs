/*macro_rules! ffmpeg {
    { $name:ident ( $( $arg:expr ),* $(,)? ) } => {
        {
            use anyhow::anyhow;
            use crate::av_strerror;
            let tmp: c_int = unsafe {
                $name($( $arg ),*)
            };
            if tmp == 0 {
                Ok(())
            }else {
                let mut cstr = Vec::with_capacity(1024);
                if unsafe { av_strerror(tmp, cstr.as_mut_ptr()as *mut c_char, 1024) } < 0 {
                    Err(
                        anyhow!(
                            "Error when calling native function `{}` in {}:{}, error code: {}.",
                            stringify!($name), file!(), line!() + 1, tmp
                        )
                    )
                }else {
                    let cstr = unsafe {
                        CStr::from_ptr(cstr.as_ptr())
                    };
                    Err(
                        anyhow!(
                            "Error when calling native function `{}` in {}:{}, error code: {}, error message: {}.",
                            stringify!($name), file!(), line!() + 1,
                            tmp, cstr.to_string_lossy()
                        )
                    )
                }
            }
        }
    };
    { $name:ident ( $( $arg:expr ),* $(,)? ) or $error_message:expr } => {
        {
            use anyhow::anyhow;

            let tmp: c_int = unsafe {
                $name($( $arg ),*)
            };
            if tmp == 0 {
                Ok(())
            }else {
                let mut cstr = Vec::with_capacity(1024);
                if unsafe { av_strerror(tmp, cstr.as_mut_ptr()as *mut c_char, 1024) } < 0 {
                    Err(
                        anyhow!(
                            "{}.\nError message:\nError when calling native function `{}` in {}:{}, error code: {}.",
                            $error_message, stringify!($name),
                            file!(), line!() + 1,
                            tmp
                        )
                    )
                }else {
                    let cstr = unsafe {
                        CStr::from_ptr(cstr.as_ptr())
                    };
                    Err(
                        anyhow!(
                            "{}.\nError message:\nError when calling native function `{}` in {}:{}, error code: {}, error message: {}.",
                            $error_message, stringify!($name),
                            file!(), line!() + 1, tmp,
                            cstr.to_string_lossy()
                        )
                    )
                }
            }
        }
    }
}*/
macro_rules! ffmpeg {
    { $name:ident ( $( $arg:expr ),* $(,)? ) } => {
        {
            use anyhow::bail;
            use crate::ffi::av_strerror;
            use std::ffi::{c_char, c_int, CStr};

            let tmp: c_int = unsafe {
                $name($( $arg ),*)
            };
            if tmp != 0 {
                let mut cstr = Vec::with_capacity(1024);
                if unsafe { av_strerror(tmp, cstr.as_mut_ptr()as *mut c_char, 1024) } < 0 {
                    bail!(
                        "Error when calling native function `{}` in {}:{}, error code: {}.",
                        stringify!($name), file!(), line!() + 1, tmp
                    );
                }else {
                    let cstr = unsafe {
                        CStr::from_ptr(cstr.as_ptr())
                    };
                    bail!(
                        "Error when calling native function `{}` in {}:{}, error code: {}, error message: {}.",
                        stringify!($name), file!(),
                        line!() + 1, tmp,
                        cstr.to_string_lossy()
                    );
                }
            }
        }
    };
    { $name:ident ( $( $arg:expr ),* $(,)? ) or $error_message:expr } => {
        {
            use anyhow::bail;
            use crate::ffi::av_strerror;
            use std::ffi::{c_char, c_int, CStr};

            let tmp: c_int = unsafe {
                $name($( $arg ),*)
            };
            if tmp != 0 {
                let mut cstr = Vec::with_capacity(1024);
                if unsafe { av_strerror(tmp, cstr.as_mut_ptr()as *mut c_char, 1024) } < 0 {
                    bail!(
                        "{}.\nError message:\nError when calling native function `{}` in {}:{}, error code: {}.",
                        $error_message, stringify!($name),
                        file!(), line!() + 1,
                        tmp
                    );
                }else {
                    let cstr = unsafe {
                        CStr::from_ptr(cstr.as_ptr())
                    };
                    bail!(
                        "{}.\nError message:\nError when calling native function `{}` in {}:{}, error code: {}, error message: {}.",
                        $error_message, stringify!($name),
                        file!(), line!() + 1, tmp,
                        cstr.to_string_lossy()
                    );
                }
            }
        }
    }
}

macro_rules! native {
    { $name:ident ( $( $arg:expr ),* $(,)? ) } => {
        {
            use anyhow::bail;
            let tmp: std::ffi::c_int = unsafe {
                $name($( $arg ),*)
            };
            if tmp < 0 {
                bail!(
                    "Error when calling native function `{}` in {}:{}, error code: {}.",
                    stringify!($name), file!(), line!() + 1, tmp
                );
            };
            tmp
        }
    };
    { $name:ident ( $( $arg:expr ),* $(,)? ) or $error_message:expr } => {
        {
            use anyhow::bail;

            let tmp: std::ffi::c_int = unsafe {
                $name($( $arg ),*)
            };
            if tmp < 0 {
                bail!(
                    "{}.\nError message:\nError when calling native function `{}` in {}:{}, error code: {}.",
                    $error_message, stringify!($name),
                    file!(), line!() + 1,
                    tmp
                );
            };
            tmp
        }
    }
}
