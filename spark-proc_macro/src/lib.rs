#![feature(proc_macro_quote)]
#![feature(let_chains)]
#![cfg_attr(debug_assertions, allow(warnings))]

use proc_macro::TokenStream;

mod clone_derive;
mod keyword;
mod native_wrapper;

#[proc_macro_derive(CloneFrom)]
pub fn clone_from(token_stream: TokenStream) -> TokenStream {
    clone_derive::clone_from(token_stream)
}

#[proc_macro]
pub fn wrap_ffmpeg(token_stream: TokenStream) -> TokenStream {
    native_wrapper::wrap_ffmpeg(token_stream)
}
