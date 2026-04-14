//! GPU attach: flush pending texture allocations and replay queued IPC payloads.

use std::sync::Arc;

use crate::ipc::SharedMemoryAccessor;
use crate::shared::{MeshUploadData, SetTexture2DData};

use super::super::AssetTransferQueue;
use super::allocations::{
    flush_pending_render_texture_allocations, flush_pending_texture_allocations,
};
use super::mesh::try_mesh_upload_with_device;
use super::texture2d::try_texture_upload_with_device;

/// After GPU [`crate::backend::RenderBackend::attach`], allocate textures for pending
/// formats and replay queued mesh/texture payloads when shared memory is available.
pub fn attach_flush_pending_asset_uploads(
    queue: &mut AssetTransferQueue,
    device: &Arc<wgpu::Device>,
    shm: Option<&mut SharedMemoryAccessor>,
) {
    flush_pending_texture_allocations(queue, device);
    flush_pending_render_texture_allocations(queue, device);
    let pending_tex: Vec<SetTexture2DData> = queue.pending_texture_uploads.drain(..).collect();
    let pending_mesh: Vec<MeshUploadData> = queue.pending_mesh_uploads.drain(..).collect();
    if let Some(shm) = shm {
        for data in pending_tex {
            try_texture_upload_with_device(queue, data, shm, None, false);
        }
        for data in pending_mesh {
            try_mesh_upload_with_device(queue, device, data, shm, None, false);
        }
    } else {
        for data in pending_tex {
            queue.pending_texture_uploads.push_back(data);
        }
        for data in pending_mesh {
            queue.pending_mesh_uploads.push_back(data);
        }
    }
}
