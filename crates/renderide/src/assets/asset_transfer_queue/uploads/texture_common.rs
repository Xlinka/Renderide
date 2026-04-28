//! Shared admission helpers for texture-family upload IPC handlers.

use std::sync::Arc;

use super::super::AssetTransferQueue;

/// Configuration for [`admit_texture_upload_data`].
pub(super) struct TextureUploadAdmission<
    'a,
    D,
    HasFormat,
    PendingLen,
    PushPending,
    HasResident,
    Flush,
> where
    HasFormat: Fn(&AssetTransferQueue, i32) -> bool,
    PendingLen: Fn(&AssetTransferQueue) -> usize,
    PushPending: Fn(&mut AssetTransferQueue, D),
    HasResident: Fn(&AssetTransferQueue, i32) -> bool,
    Flush: Fn(&mut AssetTransferQueue, &Arc<wgpu::Device>),
{
    /// Asset queue receiving the upload or deferral.
    pub(super) queue: &'a mut AssetTransferQueue,
    /// Upload command being admitted.
    pub(super) data: D,
    /// Host asset id from the upload command.
    pub(super) asset_id: i32,
    /// Payload length from the upload command's shared-memory descriptor.
    pub(super) payload_len: i32,
    /// Diagnostic asset family label.
    pub(super) kind: &'static str,
    /// Name of the format command expected before data arrives.
    pub(super) format_command: &'static str,
    /// Maximum number of deferred upload commands for this family.
    pub(super) max_pending: usize,
    /// Whether a format row is known for `asset_id`.
    pub(super) has_format: HasFormat,
    /// Current deferred upload queue length.
    pub(super) pending_len: PendingLen,
    /// Pushes `data` into the deferred upload queue.
    pub(super) push_pending: PushPending,
    /// Whether the resident GPU texture already exists.
    pub(super) has_resident: HasResident,
    /// Attempts to allocate missing textures from pending format rows.
    pub(super) flush_allocations: Flush,
}

/// Returns `Some(data)` when the texture upload can be enqueued immediately.
///
/// Empty payloads are ignored, missing formats are dropped with a warning, and uploads are
/// deferred when the GPU device/queue or resident texture is not ready yet.
pub(super) fn admit_texture_upload_data<D, HasFormat, PendingLen, PushPending, HasResident, Flush>(
    admission: TextureUploadAdmission<
        '_,
        D,
        HasFormat,
        PendingLen,
        PushPending,
        HasResident,
        Flush,
    >,
) -> Option<D>
where
    HasFormat: Fn(&AssetTransferQueue, i32) -> bool,
    PendingLen: Fn(&AssetTransferQueue) -> usize,
    PushPending: Fn(&mut AssetTransferQueue, D),
    HasResident: Fn(&AssetTransferQueue, i32) -> bool,
    Flush: Fn(&mut AssetTransferQueue, &Arc<wgpu::Device>),
{
    let TextureUploadAdmission {
        queue,
        data,
        asset_id,
        payload_len,
        kind,
        format_command,
        max_pending,
        has_format,
        pending_len,
        push_pending,
        has_resident,
        flush_allocations,
    } = admission;

    if payload_len <= 0 {
        return None;
    }
    if !has_format(queue, asset_id) {
        logger::warn!("{kind} {asset_id}: {format_command} before format; ignored");
        return None;
    }
    if queue.gpu_device.is_none() || queue.gpu_queue.is_none() {
        if pending_len(queue) >= max_pending {
            logger::warn!("{kind} {asset_id}: pending upload queue full; dropping");
            return None;
        }
        push_pending(queue, data);
        return None;
    }
    let device = queue.gpu_device.clone()?;
    if !has_resident(queue, asset_id) {
        flush_allocations(queue, &device);
    }
    if !has_resident(queue, asset_id) {
        if pending_len(queue) >= max_pending {
            logger::warn!("{kind} {asset_id}: no GPU texture and pending full; dropping data");
            return None;
        }
        push_pending(queue, data);
        return None;
    }

    Some(data)
}
