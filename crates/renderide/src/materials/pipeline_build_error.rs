//! Errors when building reflective raster mesh pipelines from composed embedded WGSL.

use thiserror::Error;

use super::wgsl_reflect::ReflectError;

/// Failure to load embedded WGSL, reflect bind layouts, or validate the per-draw contract.
#[derive(Debug, Error)]
pub enum PipelineBuildError {
    /// Naga parse/validate or bind layout rules failed for the shader source.
    #[error(transparent)]
    Reflect(#[from] ReflectError),
    /// No embedded `shaders/target/{stem}.wgsl` string in the build output for `stem`.
    #[error("embedded WGSL missing for composed stem `{0}` (run build with shaders/source)")]
    MissingEmbeddedShader(String),
}
