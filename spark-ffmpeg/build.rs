use bindgen::FieldVisibilityKind;
use std::env;
use std::path::PathBuf;
use bindgen::callbacks::{DeriveInfo, ParseCallbacks};

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
}
