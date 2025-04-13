fn main() {
    println!("cargo:rustc-link-search=/root/sherpa-gpu/lib");

    // Tell cargo to tell rustc to link the tts
    // shared library.
    println!("cargo:rustc-link-lib=tts");
}
