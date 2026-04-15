//! Command encoding for blendshape and skinning compute dispatches.

use std::num::NonZeroU64;

use glam::Mat4;

use crate::backend::advance_slab_cursor;
use crate::backend::mesh_deform::plan_blendshape_scatter_chunks;
use crate::gpu::GpuLimits;
use crate::render_graph::skinning_palette::build_skinning_palette;
use crate::scene::RenderSpaceId;

use super::snapshot::{
    deform_needs_blend_snapshot, deform_needs_skin_snapshot, MeshDeformSnapshot,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn record_mesh_deform(
    queue: &wgpu::Queue,
    device: &wgpu::Device,
    gpu_limits: &GpuLimits,
    encoder: &mut wgpu::CommandEncoder,
    pre: &crate::backend::mesh_deform::MeshPreprocessPipelines,
    scratch: &mut crate::backend::MeshDeformScratch,
    scene: &crate::scene::SceneCoordinator,
    space_id: RenderSpaceId,
    mesh: &MeshDeformSnapshot,
    bone_transform_indices: Option<&[i32]>,
    smr_node_id: i32,
    render_context: crate::shared::RenderingContext,
    head_output_transform: Mat4,
    blend_weights: &[f32],
    bone_cursor: &mut u64,
    blend_weight_cursor: &mut u64,
) {
    let Some(deform_guard) =
        validate_deform_preconditions(mesh, bone_transform_indices, gpu_limits)
    else {
        return;
    };

    if deform_guard.needs_blend {
        record_blendshape_deform(
            queue,
            device,
            gpu_limits,
            encoder,
            pre,
            scratch,
            mesh,
            blend_weights,
            blend_weight_cursor,
        );
    }

    if deform_guard.needs_skin {
        record_skinning_deform(
            queue,
            device,
            encoder,
            pre,
            scratch,
            scene,
            space_id,
            mesh,
            bone_transform_indices,
            smr_node_id,
            render_context,
            head_output_transform,
            bone_cursor,
            deform_guard.needs_blend,
            deform_guard.skin_wg,
        );
    }
}

/// Early-out state for [`record_mesh_deform`].
struct DeformValidate {
    needs_blend: bool,
    needs_skin: bool,
    /// Workgroups for skinning (`mesh_skinning.wgsl`), one thread per vertex.
    skin_wg: u32,
}

/// Returns `None` when there is no deform work or dispatch would exceed GPU limits.
fn validate_deform_preconditions(
    mesh: &MeshDeformSnapshot,
    bone_transform_indices: Option<&[i32]>,
    gpu_limits: &GpuLimits,
) -> Option<DeformValidate> {
    mesh.positions_buffer.as_ref()?;
    let vc = mesh.vertex_count;
    if vc == 0 {
        return None;
    }
    let needs_blend = deform_needs_blend_snapshot(mesh);
    let needs_skin = deform_needs_skin_snapshot(mesh, bone_transform_indices);

    if !needs_blend && !needs_skin {
        return None;
    }

    let skin_wg = workgroup_count(vc);
    if needs_skin && !gpu_limits.compute_dispatch_fits(skin_wg, 1, 1) {
        logger::warn!(
            "mesh deform: skinning dispatch {}×1×1 exceeds max_compute_workgroups_per_dimension ({})",
            skin_wg,
            gpu_limits.max_compute_workgroups_per_dimension()
        );
        return None;
    }

    Some(DeformValidate {
        needs_blend,
        needs_skin,
        skin_wg,
    })
}

/// Sparse blendshape scatter: copy bind poses → temp, then one scatter dispatch per weighted shape chunk.
#[allow(clippy::too_many_arguments)]
fn record_blendshape_deform(
    queue: &wgpu::Queue,
    device: &wgpu::Device,
    gpu_limits: &GpuLimits,
    encoder: &mut wgpu::CommandEncoder,
    pre: &crate::backend::mesh_deform::MeshPreprocessPipelines,
    scratch: &mut crate::backend::MeshDeformScratch,
    mesh: &MeshDeformSnapshot,
    blend_weights: &[f32],
    blend_weight_cursor: &mut u64,
) {
    let Some(ref positions) = mesh.positions_buffer else {
        return;
    };
    let Some(ref temp) = mesh.deform_temp_buffer else {
        return;
    };
    let Some(ref sparse) = mesh.blendshape_sparse_buffer else {
        return;
    };
    let vc = mesh.vertex_count;
    let shape_count = mesh.num_blendshapes;
    if shape_count == 0 {
        return;
    }
    if mesh.blendshape_sparse_ranges.len() != shape_count as usize {
        logger::warn!(
            "mesh deform: blendshape_sparse_ranges len {} != num_blendshapes {}",
            mesh.blendshape_sparse_ranges.len(),
            shape_count
        );
        return;
    }

    let copy_len = u64::from(vc).saturating_mul(16).max(16);
    encoder.copy_buffer_to_buffer(positions.as_ref(), 0, temp.as_ref(), 0, copy_len);

    scratch.ensure_shape_weight_capacity(device, shape_count);
    let mut wbytes = vec![0u8; (shape_count as usize) * 4];
    for s in 0..shape_count as usize {
        let w = blend_weights.get(s).copied().unwrap_or(0.0);
        wbytes[s * 4..s * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }

    let weight_binding_len = wbytes.len() as u64;
    scratch.ensure_blend_weight_byte_capacity(
        device,
        (*blend_weight_cursor).saturating_add(weight_binding_len),
    );
    queue.write_buffer(&scratch.blendshape_weights, *blend_weight_cursor, &wbytes);

    let max_wg = gpu_limits.max_compute_workgroups_per_dimension();

    let mut packed_params: Vec<u8> = Vec::new();
    let mut dispatch_wgs: Vec<u32> = Vec::new();

    for s in 0..shape_count {
        let w = blend_weights.get(s as usize).copied().unwrap_or(0.0);
        if w == 0.0 {
            continue;
        }
        let (first, cnt) = mesh.blendshape_sparse_ranges[s as usize];
        if cnt == 0 {
            continue;
        }
        for (sparse_base, sparse_count) in plan_blendshape_scatter_chunks(first, cnt, max_wg) {
            let wg = workgroup_count(sparse_count);
            if !gpu_limits.compute_dispatch_fits(wg, 1, 1) {
                logger::warn!(
                    "mesh deform: blendshape scatter dispatch {}×1×1 exceeds max_compute_workgroups_per_dimension ({})",
                    wg,
                    max_wg
                );
                return;
            }
            packed_params.extend_from_slice(&build_scatter_params(
                vc,
                s,
                sparse_base,
                sparse_count,
            ));
            dispatch_wgs.push(wg);
        }
    }

    if packed_params.is_empty() {
        *blend_weight_cursor = advance_slab_cursor(*blend_weight_cursor, weight_binding_len);
        return;
    }

    scratch.ensure_blendshape_params_staging(device, packed_params.len() as u64);
    queue.write_buffer(&scratch.blendshape_params_staging, 0, &packed_params);

    let Some(weight_size) = NonZeroU64::new(weight_binding_len) else {
        *blend_weight_cursor = advance_slab_cursor(*blend_weight_cursor, weight_binding_len);
        return;
    };

    for (i, &scatter_wg) in dispatch_wgs.iter().enumerate() {
        let src_off = (i as u64).saturating_mul(32);
        encoder.copy_buffer_to_buffer(
            &scratch.blendshape_params_staging,
            src_off,
            &scratch.blendshape_params,
            0,
            32,
        );

        let blend_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blendshape_scatter_bg"),
            layout: &pre.blendshape_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scratch.blendshape_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sparse.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &scratch.blendshape_weights,
                        offset: *blend_weight_cursor,
                        size: Some(weight_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: temp.as_entire_binding(),
                },
            ],
        });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("blendshape_scatter"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pre.blendshape_pipeline);
        cpass.set_bind_group(0, &blend_bg, &[]);
        cpass.dispatch_workgroups(scatter_wg, 1, 1);
    }

    *blend_weight_cursor = advance_slab_cursor(*blend_weight_cursor, weight_binding_len);
}

/// Linear blend skinning compute after optional blendshape pass.
#[allow(clippy::too_many_arguments)]
fn record_skinning_deform(
    queue: &wgpu::Queue,
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pre: &crate::backend::mesh_deform::MeshPreprocessPipelines,
    scratch: &mut crate::backend::MeshDeformScratch,
    scene: &crate::scene::SceneCoordinator,
    space_id: RenderSpaceId,
    mesh: &MeshDeformSnapshot,
    bone_transform_indices: Option<&[i32]>,
    smr_node_id: i32,
    render_context: crate::shared::RenderingContext,
    head_output_transform: Mat4,
    bone_cursor: &mut u64,
    needs_blend: bool,
    wg: u32,
) {
    let Some(ref positions) = mesh.positions_buffer else {
        return;
    };
    let Some(ref dst) = mesh.deformed_positions_buffer else {
        return;
    };
    let Some(ref dst_n) = mesh.deformed_normals_buffer else {
        return;
    };
    let Some(ref src_n) = mesh.normals_buffer else {
        return;
    };
    let Some(ref bone_idx) = mesh.bone_indices_buffer else {
        return;
    };
    let Some(ref bone_wt) = mesh.bone_weights_vec4_buffer else {
        return;
    };
    let Some(indices) = bone_transform_indices else {
        return;
    };

    let bone_count_u = mesh.skinning_bind_matrices.len() as u32;
    scratch.ensure_bone_capacity(device, bone_count_u);
    let Some(palette_mats) = build_skinning_palette(
        scene,
        space_id,
        &mesh.skinning_bind_matrices,
        mesh.has_skeleton,
        indices,
        smr_node_id,
        render_context,
        head_output_transform,
    ) else {
        return;
    };
    let mut palette: Vec<u8> = vec![0u8; palette_mats.len() * 64];
    for (bi, pal) in palette_mats.iter().enumerate() {
        let cols = pal.to_cols_array();
        palette[bi * 64..bi * 64 + 64].copy_from_slice(bytemuck::cast_slice(&cols));
    }

    let palette_len = palette.len() as u64;
    scratch.ensure_bone_byte_capacity(device, bone_cursor.saturating_add(palette_len));
    queue.write_buffer(&scratch.bone_matrices, *bone_cursor, &palette);

    let Some(bone_binding_size) = NonZeroU64::new(palette_len) else {
        return;
    };

    let src_for_skin: &wgpu::Buffer = if needs_blend {
        let Some(buf) = mesh.deform_temp_buffer.as_deref() else {
            return;
        };
        buf
    } else {
        positions.as_ref()
    };

    skinning_dispatch_with_uploaded_palette(
        device,
        encoder,
        pre,
        scratch,
        src_for_skin,
        bone_idx,
        bone_wt,
        dst,
        src_n,
        dst_n,
        *bone_cursor,
        bone_binding_size,
        wg,
    );

    *bone_cursor = advance_slab_cursor(*bone_cursor, palette_len);
}

/// Builds skinning bind group (bone slab + attributes) and dispatches the skinning shader.
#[allow(clippy::too_many_arguments)]
fn skinning_dispatch_with_uploaded_palette(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pre: &crate::backend::mesh_deform::MeshPreprocessPipelines,
    scratch: &crate::backend::MeshDeformScratch,
    src_positions: &wgpu::Buffer,
    bone_idx: &wgpu::Buffer,
    bone_wt: &wgpu::Buffer,
    dst_pos: &wgpu::Buffer,
    src_n: &wgpu::Buffer,
    dst_n: &wgpu::Buffer,
    bone_cursor: u64,
    bone_binding_size: NonZeroU64,
    wg: u32,
) {
    let skin_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("skinning_bg"),
        layout: &pre.skinning_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &scratch.bone_matrices,
                    offset: bone_cursor,
                    size: Some(bone_binding_size),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: src_positions.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: bone_idx.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: bone_wt.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: dst_pos.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: src_n.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: dst_n.as_entire_binding(),
            },
        ],
    });

    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("skinning"),
        timestamp_writes: None,
    });
    cpass.set_pipeline(&pre.skinning_pipeline);
    cpass.set_bind_group(0, &skin_bg, &[]);
    cpass.dispatch_workgroups(wg, 1, 1);
}

fn workgroup_count(count: u32) -> u32 {
    (count.saturating_add(63)) / 64
}

/// Encodes `source/compute/mesh_blendshape.wgsl` `Params` (32 bytes).
fn build_scatter_params(
    vertex_count: u32,
    shape_index: u32,
    sparse_base: u32,
    sparse_count: u32,
) -> [u8; 32] {
    let mut o = [0u8; 32];
    o[0..4].copy_from_slice(&vertex_count.to_le_bytes());
    o[4..8].copy_from_slice(&shape_index.to_le_bytes());
    o[8..12].copy_from_slice(&sparse_base.to_le_bytes());
    o[12..16].copy_from_slice(&sparse_count.to_le_bytes());
    o
}
