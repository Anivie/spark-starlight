use crate::ffi::avio_alloc_context;
use anyhow::anyhow;
use spark_proc_macro::wrap_ffmpeg;
use std::os::raw::{c_int, c_void};

wrap_ffmpeg!(
    AVIOContext drop+ [avio_context_free]
);

pub type ReadPacketCallback =
    unsafe extern "C" fn(opaque: *mut c_void, buf: *mut u8, buf_size: c_int) -> c_int;

pub type WritePacketCallback =
    unsafe extern "C" fn(opaque: *mut c_void, buf: *const u8, buf_size: c_int) -> c_int;

pub type SeekCallback =
    unsafe extern "C" fn(opaque: *mut c_void, offset: i64, whence: c_int) -> i64;

impl AVIOContext {
    /*avio_alloc_context(
        (unsigned char*)av_malloc(image_size + AV_INPUT_BUFFER_PADDING_SIZE),
        image_size,  // 缓冲区大小
        0,           // 不可写
        NULL,        // 不透明指针
        NULL,        // 读回调(不使用)
        NULL,        // 写回调(不使用)
        NULL         // seek回调(不使用)
    )*/

    pub fn alloc(
        buffer: *mut u8,
        buffer_size: usize,
        write_flag: i32,
        opaque: *mut std::ffi::c_void,
        read_packet: Option<ReadPacketCallback>,
        write_packet: Option<WritePacketCallback>,
        seek: Option<SeekCallback>,
    ) -> anyhow::Result<AVIOContext> {
        unsafe {
            let context = avio_alloc_context(
                buffer,
                buffer_size as i32,
                write_flag,
                opaque,
                read_packet,
                write_packet,
                seek,
            );

            if context.is_null() {
                return Err(anyhow!("Failed to allocate AVIOContext"));
            }

            Ok(AVIOContext { inner: context })
        }
    }

    pub fn fill_data(&mut self, from: &[u8]) -> anyhow::Result<()> {
        unsafe {
            let buffer = (*self.inner).buffer;

            // Copy the data into the buffer
            std::ptr::copy_nonoverlapping(from.as_ptr(), buffer, from.len());

            // Set the buffer size
            (*self.inner).buffer_size = from.len() as i32;

            Ok(())
        }
    }
}
