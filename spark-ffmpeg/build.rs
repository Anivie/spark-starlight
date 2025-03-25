#![feature(substr_range)]
#![feature(let_chains)]

use bindgen::callbacks::{DeriveInfo, ParseCallbacks};
use bindgen::FieldVisibilityKind;
use convert_case::{Case, Casing};
use rayon::prelude::*;
use std::collections::HashMap;
use std::env;
use std::path::PathBuf;

#[derive(Debug)]
struct Cat;

impl ParseCallbacks for Cat {
    fn add_derives(&self, info: &DeriveInfo<'_>) -> Vec<String> {
        if info.kind == bindgen::callbacks::TypeKind::Struct {
            vec!["CloneFrom".into()]
        } else {
            vec![]
        }
    }
}

fn main() {
    // Tell cargo to look for shared libraries in the specified directory
    //println!("cargo:rustc-link-search=/home/anivie/SDK/ffmpeg/lib");
    println!("cargo:rustc-link-search=/root/ffmpeg/lib");

    // Tell cargo to tell rustc to link the system ffmpeg
    // shared library.
    println!("cargo:rustc-link-lib=avcodec");
    println!("cargo:rustc-link-lib=avformat");
    println!("cargo:rustc-link-lib=avutil");
    println!("cargo:rustc-link-lib=swscale");
    println!("cargo:rustc-link-lib=avfilter");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("./ffi/ffmpeg.h")
        //.clang_arg("-I/home/anivie/SDK/ffmpeg/include")
        .clang_arg("-I/root/ffmpeg/include")
        .generate_comments(false)
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .raw_line("use spark_proc_macro::CloneFrom;")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .derive_default(true)
        .blocklist_item("FP_NAN")
        .blocklist_item("FP_INFINITE")
        .blocklist_item("FP_ZERO")
        .blocklist_item("FP_SUBNORMAL")
        .blocklist_item("FP_NORMAL")
        .default_visibility(FieldVisibilityKind::PublicCrate)
        .merge_extern_blocks(true)
        .parse_callbacks(Box::new(Cat))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    let codec_id = extract_enum(&bindings.to_string(), "AVCodecID_AV_CODEC_ID_", "u32");
    std::fs::write(out_path.join("codec_id.rs"), codec_id).unwrap();

    let codec_id = extract_enum(&bindings.to_string(), "AVPixelFormat_AV_PIX_FMT_", "i32");
    std::fs::write(out_path.join("pixel.rs"), codec_id).unwrap();
}

fn extract_enum(target: &str, pattern: &str, types: &str) -> String {
    let name = pattern
        .chars()
        .take_while(|x| *x != '_')
        .collect::<String>();
    let middle = target
        .split('\n')
        .par_bridge()
        .filter(|x| x.contains(pattern))
        .map(|x| {
            let string = x.chars().skip(10).skip(pattern.len()).collect::<String>();
            let string = string
                .replace(format!(": {}", name).as_str(), "")
                .replace(";", ",\n")
                .to_case(Case::UpperCamel);
            let string = if let Some(c) = string.chars().next()
                && c.is_ascii_digit()
            {
                format!("_{}", string)
            } else {
                string
            };

            let tail = string
                .chars()
                .rev()
                .take_while(|x| *x != '=')
                .collect::<String>();
            let tail = tail.chars().rev().collect::<String>();

            (tail, format!("    {}", string))
        })
        .collect::<HashMap<String, String>>();

    let middle = middle.values().cloned().collect::<String>();

    format!("#[repr({})]\n#[derive(num_enum::TryFromPrimitive, Copy, Clone, PartialEq, Eq, Debug, Hash)]\npub enum {} {{\n{}\n}}", types, name, middle)
}
