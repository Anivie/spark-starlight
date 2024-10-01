#![feature(proc_macro_quote)]
#![cfg_attr(debug_assertions, allow(warnings))]

use proc_macro::TokenStream;

mod keyword;
mod native_wrapper;

#[proc_macro]
pub fn wrap_ffmpeg(token_stream: TokenStream) -> TokenStream {
    native_wrapper::wrap_ffmpeg(token_stream)
}