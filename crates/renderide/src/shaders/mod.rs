//! Build-time converted Unity/Resonite shaders and generated material descriptors.
//!
//! **UnityShaderConverter** writes one directory per shader under this folder (for example
//! [`converter_minimal_unlit`](converter_minimal_unlit/): `mod.rs`, `material.rs`, `passN.wgsl`).
//! There is no `generated/` re-export tree; add `pub mod <stem>;` here for each converted shader.
//!
//! Regenerate from the `Renderide/` directory:
//! `dotnet run --project generators/UnityShaderConverter --` (with `slangc` on `PATH` / `SLANGC`), or `--skip-slang` if WGSL is already on disk.

pub mod converter_minimal_unlit;

#[cfg(test)]
mod wgsl_validate_tests {
    /// Validates that committed sample WGSL parses with the same `naga` revision `wgpu` uses.
    #[test]
    fn minimal_unlit_sample_wgsl_parses() {
        let src = super::converter_minimal_unlit::material::PASS0_WGSL;
        let mut front = naga::front::wgsl::Frontend::new();
        let _ = front
            .parse(src)
            .expect("sample WGSL should parse; fix the file or regenerate with slangc");
    }
}
