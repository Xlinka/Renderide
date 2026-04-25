//! Build script that embeds a Windows side-by-side manifest declaring a
//! dependency on Common Controls v6, so `rfd`'s `TaskDialogIndirect` import
//! resolves at process load time instead of failing with "Entry Point Not
//! Found in comctl32.dll".
//!
//! `embed-manifest` only emits `cargo:rustc-link-arg-bins=...`, which Cargo
//! applies exclusively to `[[bin]]` targets, so the unit and integration test
//! executables produced by `cargo test` would otherwise be unmanifested. Since
//! `vr_prompt` (a `pub mod` of the library) statically imports `TaskDialogIndirect`
//! via `rfd`'s `common-controls-v6` feature, those test executables aborted at
//! process load with `STATUS_ENTRYPOINT_NOT_FOUND` (0xc0000139) on Windows CI.
//! After calling `embed-manifest`, this script re-emits the equivalent linker
//! arguments under `cargo:rustc-link-arg-tests=...`, reusing the manifest
//! artifact `embed-manifest` already wrote into `OUT_DIR`.

use std::path::PathBuf;

/// Cargo build script entry point: embeds the Common Controls v6 manifest into
/// the bootstrapper binary and its test executables on Windows targets, and is
/// a no-op on other platforms.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_CFG_WINDOWS").is_none() {
        return Ok(());
    }

    use embed_manifest::{embed_manifest, new_manifest};
    embed_manifest(new_manifest("Renderide.Bootstrapper"))?;

    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").ok_or("OUT_DIR is not set")?);
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    match target_env.as_str() {
        "msvc" => {
            let manifest_path = out_dir.join("manifest.xml").canonicalize()?;
            println!("cargo:rustc-link-arg-tests=/MANIFEST:EMBED");
            println!(
                "cargo:rustc-link-arg-tests=/MANIFESTINPUT:{}",
                manifest_path.display()
            );
            println!("cargo:rustc-link-arg-tests=/MANIFESTUAC:NO");
        }
        "gnu" => {
            let object_path = out_dir.join("embed-manifest.o");
            println!("cargo:rustc-link-arg-tests={}", object_path.display());
        }
        _ => {}
    }
    Ok(())
}
