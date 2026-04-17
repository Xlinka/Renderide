//! Main forward pass: clear color + depth, debug normal shading for scene meshes.
//!
//! Draws are collected and **sorted by [`MaterialDrawBatchKey`](crate::render_graph::MaterialDrawBatchKey)**
//! so pipeline and batch key drive pipeline switches. **GPU instancing:** consecutive draws that share the
//! same mesh submesh and batch key (opaque, non-skinned) are merged into one indexed draw with
//! `instance_index` sampling [`crate::backend::PerDrawResources`] (`@group(2)` storage). Embedded `@group(1)`
//! skips redundant [`wgpu::RenderPass::set_bind_group`] when [`MaterialBindCacheKey`](crate::backend::MaterialBindCacheKey) matches
//! the previous draw (uniform updates still run each time via [`EmbeddedMaterialBindResources`](crate::backend::EmbeddedMaterialBindResources)).
//! Per-slot [`MaterialPropertyLookupIds`](crate::assets::material::MaterialPropertyLookupIds) are carried on each
//! [`WorldMeshDrawItem`](crate::render_graph::WorldMeshDrawItem) for `get_merged` when building `@group(1)` bind
//! groups for [`crate::materials::RasterPipelineKind::EmbeddedStem`] draws (see [`crate::backend::EmbeddedMaterialBindResources`]).
//!
//! Manifest raster binds use the composed WGSL **stem** from [`crate::materials::MaterialRouter::stem_for_shader_asset`]
//! (not a hard-coded Unlit path). Whether UV0 is bound is stored on [`MaterialDrawBatchKey::Embedded_needs_uv0`]
//! (same rule as the embedded raster pipeline and [`crate::materials::embedded_stem_needs_uv0_stream`], computed during draw collection).
//! Intersection tint subpasses use [`MaterialDrawBatchKey::Embedded_requires_intersection_pass`]
//! ([`crate::materials::embedded_stem_requires_intersection_pass`], WGSL reflection of `_IntersectColor`).
//!
//! ## VR stereo world draws
//!
//! OpenXR per-eye view–projection maps **stage** space to clip. For **non-overlay** draws with
//! `stereo_view_proj`, we use **identity** instead of the host `view_transform` world-to-camera so
//! `VP` is not `P·V_hmd·V_host`, which mixed stage with the host rig and caused playspace-relative
//! offsets. Overlays keep `view` for orthographic / UI alignment with the host camera rig.
//! Matrix composition lives in [`vp`].

mod encode;
mod execute_helpers;
mod vp;

use crate::render_graph::context::RenderPassContext;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::pass::{PassBuilder, RenderPass};
use crate::render_graph::resources::{
    BufferAccess, ImportedBufferHandle, ImportedTextureHandle, StorageAccess,
};

use execute_helpers::{
    encode_clear_only_pass, encode_msaa_depth_resolve_after_clear_only,
    encode_world_mesh_forward_draw_passes, prepare_world_mesh_forward_frame,
    ForwardPassEncodeFrame, ForwardPassEncodeViews,
};

/// Clears the backbuffer and depth, then draws meshes with material-batched raster pipelines.
#[derive(Debug)]
pub struct WorldMeshForwardPass {
    resources: WorldMeshForwardGraphResources,
}

/// Graph resources used by [`WorldMeshForwardPass`].
#[derive(Clone, Copy, Debug)]
pub struct WorldMeshForwardGraphResources {
    /// Imported frame color target.
    pub color: ImportedTextureHandle,
    /// Imported frame depth target.
    pub depth: ImportedTextureHandle,
    /// Imported cluster light-count storage buffer.
    pub cluster_light_counts: ImportedBufferHandle,
    /// Imported cluster light-index storage buffer.
    pub cluster_light_indices: ImportedBufferHandle,
    /// Imported light storage buffer.
    pub lights: ImportedBufferHandle,
    /// Imported per-draw storage slab.
    pub per_draw_slab: ImportedBufferHandle,
    /// Imported frame uniform buffer.
    pub frame_uniforms: ImportedBufferHandle,
}

impl WorldMeshForwardPass {
    /// Creates a world mesh forward pass instance.
    pub fn new(resources: WorldMeshForwardGraphResources) -> Self {
        Self { resources }
    }
}

impl RenderPass for WorldMeshForwardPass {
    fn name(&self) -> &str {
        "WorldMeshForward"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        {
            let mut r = b.raster();
            r.color(
                self.resources.color,
                wgpu::Operations {
                    load: wgpu::LoadOp::Clear(crate::present::SWAPCHAIN_CLEAR_COLOR),
                    store: wgpu::StoreOp::Store,
                },
                Option::<ImportedTextureHandle>::None,
            );
            r.depth(
                self.resources.depth,
                wgpu::Operations {
                    load: wgpu::LoadOp::Clear(crate::render_graph::MAIN_FORWARD_DEPTH_CLEAR),
                    store: wgpu::StoreOp::Store,
                },
                None,
            );
        }
        b.import_buffer(
            self.resources.cluster_light_counts,
            BufferAccess::Storage {
                stages: wgpu::ShaderStages::FRAGMENT,
                access: StorageAccess::ReadOnly,
            },
        );
        b.import_buffer(
            self.resources.cluster_light_indices,
            BufferAccess::Storage {
                stages: wgpu::ShaderStages::FRAGMENT,
                access: StorageAccess::ReadOnly,
            },
        );
        b.import_buffer(
            self.resources.lights,
            BufferAccess::Storage {
                stages: wgpu::ShaderStages::FRAGMENT,
                access: StorageAccess::ReadOnly,
            },
        );
        b.import_buffer(
            self.resources.per_draw_slab,
            BufferAccess::Storage {
                stages: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                access: StorageAccess::ReadOnly,
            },
        );
        b.import_buffer(
            self.resources.frame_uniforms,
            BufferAccess::Uniform {
                stages: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                dynamic_offset: false,
            },
        );
        Ok(())
    }

    fn execute(&mut self, ctx: &mut RenderPassContext<'_, '_, '_>) -> Result<(), RenderPassError> {
        let Some(bb) = ctx.backbuffer else {
            return Err(RenderPassError::MissingBackbuffer {
                pass: self.name().to_string(),
            });
        };
        let Some(depth) = ctx.depth_view else {
            return Err(RenderPassError::MissingDepth {
                pass: self.name().to_string(),
            });
        };
        let Some(frame) = ctx.frame.as_mut() else {
            return Err(RenderPassError::MissingFrameParams {
                pass: self.name().to_string(),
            });
        };

        let queue_guard = ctx
            .queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let queue = &*queue_guard;

        frame.prepared_world_mesh_forward =
            prepare_world_mesh_forward_frame(ctx.device, queue, ctx.gpu_limits, frame);
        let Some(prepared) = frame.prepared_world_mesh_forward.take() else {
            return Ok(());
        };

        let msaa_color = frame.msaa_color_view.clone();
        let msaa_depth = frame.msaa_depth_view.clone();
        let color_view = msaa_color.as_ref().unwrap_or(bb);
        let depth_raster = msaa_depth.as_ref().unwrap_or(depth);
        let resolve_swapchain = if frame.sample_count > 1 {
            Some(bb)
        } else {
            None
        };

        let msaa_depth_resolve = frame.backend.msaa_depth_resolve.clone();

        if prepared.draws.is_empty() {
            encode_clear_only_pass(
                ctx.encoder,
                color_view,
                depth_raster,
                prepared.pipeline.pass_desc.depth_stencil_format,
                resolve_swapchain,
                prepared.pipeline.use_multiview,
            );
            encode_msaa_depth_resolve_after_clear_only(
                ctx.device,
                ctx.encoder,
                frame,
                msaa_depth_resolve.as_deref(),
            );
            return Ok(());
        }

        if !encode_world_mesh_forward_draw_passes(
            ForwardPassEncodeFrame {
                encoder: ctx.encoder,
                device: ctx.device,
                frame,
                queue,
            },
            &prepared,
            ForwardPassEncodeViews {
                color_view,
                depth_raster_view: depth_raster,
                resolve_swapchain,
                msaa_depth_resolve: msaa_depth_resolve.as_deref(),
            },
        ) {
            return Ok(());
        }

        Ok(())
    }
}
