//! Asset ingestion, shader routing helpers, and GPU upload orchestration.
//!
//! [`asset_transfer_queue::AssetTransferQueue`] holds mesh/texture pools, IPC upload queues, and
//! CPU-side format tables; [`crate::backend::RenderBackend`] owns the live instance after
//! construction.

pub mod asset_transfer_queue;
pub mod material;
pub mod mesh;
pub mod shader;
pub mod texture;
pub mod util;

pub use asset_transfer_queue::AssetTransferQueue;
pub use shader::{
    resolve_shader_routing_name_from_upload, resolve_shader_upload, ResolvedShaderUpload,
};

/// Alias for [`AssetTransferQueue`] (same type: upload queues, pools, and per-poll budgets).
pub type AssetSubsystem = AssetTransferQueue;
