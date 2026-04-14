//! Shared-memory ingestion and queue draining for [`super::AssetTransferQueue`].
//!
//! Split into [`allocations`], [`attach`], [`texture2d`], [`mesh`], and [`render_texture`] for clarity.

mod allocations;
mod attach;
mod mesh;
mod render_texture;
mod texture2d;

pub use attach::attach_flush_pending_asset_uploads;
pub use mesh::{
    begin_ipc_poll_mesh_upload_budget, drain_deferred_mesh_uploads_after_poll, on_mesh_unload,
    try_process_mesh_upload,
};
pub use render_texture::{on_set_render_texture_format, on_unload_render_texture};
pub use texture2d::{
    drain_deferred_texture_uploads_after_poll, on_set_texture_2d_data, on_set_texture_2d_format,
    on_set_texture_2d_properties, on_unload_texture_2d, try_texture_upload_with_device,
};

/// Max queued [`MeshUploadData`](crate::shared::MeshUploadData) when GPU is not ready yet (host data stays in shared memory).
pub const MAX_PENDING_MESH_UPLOADS: usize = 256;

/// Max deferred low-priority mesh uploads when [`MESH_UPLOAD_NON_HIGH_PRIORITY_BUDGET_PER_POLL`] is hit.
pub const MAX_DEFERRED_MESH_UPLOADS: usize = 512;

/// Max deferred mesh uploads drained at the end of one [`crate::runtime::RendererRuntime::poll_ipc`]
/// (cross-tick backlog may span multiple polls).
pub const MAX_DEFERRED_MESH_UPLOADS_DRAIN_PER_POLL: usize = 64;

/// Max non-[`MeshUploadData::high_priority`](crate::shared::MeshUploadData) mesh uploads processed inline per
/// [`crate::runtime::RendererRuntime::poll_ipc`] before additional commands are deferred.
pub const MESH_UPLOAD_NON_HIGH_PRIORITY_BUDGET_PER_POLL: u32 = 32;

/// Max non-[`SetTexture2DData::high_priority`](crate::shared::SetTexture2DData) texture data uploads processed inline per
/// [`crate::runtime::RendererRuntime::poll_ipc`] before additional commands are deferred.
pub const TEXTURE_UPLOAD_NON_HIGH_PRIORITY_BUDGET_PER_POLL: u32 = 32;

/// Max deferred low-priority texture uploads when [`TEXTURE_UPLOAD_NON_HIGH_PRIORITY_BUDGET_PER_POLL`] is hit.
pub const MAX_DEFERRED_TEXTURE_UPLOADS: usize = 512;

/// Max deferred texture uploads drained at the end of one [`crate::runtime::RendererRuntime::poll_ipc`].
pub const MAX_DEFERRED_TEXTURE_UPLOADS_DRAIN_PER_POLL: usize = 64;

/// Max queued texture data commands when GPU or format is not ready.
pub const MAX_PENDING_TEXTURE_UPLOADS: usize = 256;
