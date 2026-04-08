//! Host [`ShaderUpload`](crate::shared::ShaderUpload) handling: logical name extraction and material routing.

pub mod logical_name;
pub mod route;
pub mod unity_asset;

pub use logical_name::resolve_shader_routing_name_from_upload;
pub use route::{resolve_shader_upload, ResolvedShaderUpload};
