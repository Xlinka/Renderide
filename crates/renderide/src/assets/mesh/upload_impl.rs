//! Helpers for [`super::GpuMesh::upload`](GpuMesh::upload); keeps the `impl` readable.

use std::sync::Arc;

use glam::Mat4;
use wgpu::util::DeviceExt;

use crate::backend::mesh_deform::plan_blendshape_bind_chunks;
use crate::gpu::GpuLimits;
use crate::shared::MeshUploadData;

use super::gpu_mesh_hints::wgpu_index_format;
use super::layout::{
    color_float4_stream_bytes, compute_index_count, compute_vertex_stride, extract_bind_poses,
    extract_blendshape_offsets, extract_float3_position_normal_as_vec4_streams,
    split_bone_weights_tail_for_gpu, synthetic_bone_data_for_blendshape_only,
    uv0_float2_stream_bytes, MeshBufferLayout,
};

/// Interleaved VB, IB, and layout-derived scalars after validation.
pub(super) struct CoreBuffers {
    pub vb: wgpu::Buffer,
    pub ib: wgpu::Buffer,
    pub index_format: wgpu::IndexFormat,
    pub vertex_stride: u32,
    pub vertex_stride_us: usize,
    pub index_count_u32: u32,
}

/// Aggregated bone/skin GPU state and skinning matrices.
pub(super) struct BoneSkinUpload {
    pub bone_counts_buffer: Option<Arc<wgpu::Buffer>>,
    pub bone_indices_buffer: Option<Arc<wgpu::Buffer>>,
    pub bone_weights_vec4_buffer: Option<Arc<wgpu::Buffer>>,
    pub bind_poses_buffer: Option<Arc<wgpu::Buffer>>,
    pub skinning_bind_matrices: Vec<Mat4>,
}

/// Position/normal streams, UV0, and vertex color.
pub(super) struct DerivedStreams {
    pub positions_buffer: Option<Arc<wgpu::Buffer>>,
    pub normals_buffer: Option<Arc<wgpu::Buffer>>,
    pub uv0_buffer: Option<Arc<wgpu::Buffer>>,
    pub color_buffer: Option<Arc<wgpu::Buffer>>,
}

/// Validates raw length and device max buffer size; returns `None` when upload must abort.
pub(super) fn validate_mesh_upload_layout(
    raw: &[u8],
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
    max_buf: u64,
) -> bool {
    if raw.len() < layout.total_buffer_length {
        logger::error!(
            "mesh {}: raw too short (need {}, got {})",
            data.asset_id,
            layout.total_buffer_length,
            raw.len()
        );
        return false;
    }

    if layout.vertex_size as u64 > max_buf
        || layout.index_buffer_length as u64 > max_buf
        || layout.total_buffer_length as u64 > max_buf
    {
        logger::warn!(
            "mesh {}: buffer layout exceeds max_buffer_size ({})",
            data.asset_id,
            max_buf
        );
        return false;
    }
    true
}

/// Creates core vertex and index buffers from the layout-validated `raw` slice.
pub(super) fn create_core_vertex_index_buffers(
    device: &wgpu::Device,
    raw: &[u8],
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
) -> CoreBuffers {
    let vertex_stride = compute_vertex_stride(&data.vertex_attributes).max(1) as u32;
    let vertex_stride_us = vertex_stride as usize;
    let index_count = compute_index_count(&data.submeshes);
    let index_count_u32 = index_count.max(0) as u32;

    let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("mesh {} vertices", data.asset_id)),
        contents: &raw[..layout.vertex_size],
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });

    let ib_slice =
        &raw[layout.index_buffer_start..layout.index_buffer_start + layout.index_buffer_length];
    let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("mesh {} indices", data.asset_id)),
        contents: ib_slice,
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
    });

    let index_format = wgpu_index_format(data.index_buffer_format);

    CoreBuffers {
        vb,
        ib,
        index_format,
        vertex_stride,
        vertex_stride_us,
        index_count_u32,
    }
}

fn upload_positions_normals(
    device: &wgpu::Device,
    data: &MeshUploadData,
    vertex_slice: &[u8],
    vc_usize: usize,
    vertex_stride_us: usize,
) -> (Option<Arc<wgpu::Buffer>>, Option<Arc<wgpu::Buffer>>) {
    match extract_float3_position_normal_as_vec4_streams(
        vertex_slice,
        vc_usize,
        vertex_stride_us,
        &data.vertex_attributes,
    ) {
        Some((pb, nb)) => {
            let usage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST;
            let pbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh {} positions_stream", data.asset_id)),
                contents: &pb,
                usage,
            });
            let nbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh {} normals_stream", data.asset_id)),
                contents: &nb,
                usage,
            });
            (Some(Arc::new(pbuf)), Some(Arc::new(nbuf)))
        }
        None => {
            logger::warn!(
                "mesh {}: missing float3 position+normal attributes — debug/deform path disabled",
                data.asset_id
            );
            (None, None)
        }
    }
}

fn upload_uv0_color(
    device: &wgpu::Device,
    data: &MeshUploadData,
    vertex_slice: &[u8],
    vc_usize: usize,
    vertex_stride_us: usize,
) -> (Option<Arc<wgpu::Buffer>>, Option<Arc<wgpu::Buffer>>) {
    let uv0_buffer = uv0_float2_stream_bytes(
        vertex_slice,
        vc_usize,
        vertex_stride_us,
        &data.vertex_attributes,
    )
    .map(|uv_bytes| {
        Arc::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh {} uv0_stream", data.asset_id)),
                contents: &uv_bytes,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }),
        )
    });
    let color_buffer = color_float4_stream_bytes(
        vertex_slice,
        vc_usize,
        vertex_stride_us,
        &data.vertex_attributes,
    )
    .map(|color_bytes| {
        Arc::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh {} color_stream", data.asset_id)),
                contents: &color_bytes,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }),
        )
    });
    (uv0_buffer, color_buffer)
}

/// Builds optional position/normal streams plus UV0 and vertex color buffers.
pub(super) fn extract_derived_vertex_streams(
    device: &wgpu::Device,
    raw: &[u8],
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
    core: &CoreBuffers,
) -> DerivedStreams {
    let vc_usize = data.vertex_count.max(0) as usize;
    let vertex_slice = &raw[..layout.vertex_size];
    let (positions_buffer, normals_buffer) =
        upload_positions_normals(device, data, vertex_slice, vc_usize, core.vertex_stride_us);
    let (uv0_buffer, color_buffer) =
        upload_uv0_color(device, data, vertex_slice, vc_usize, core.vertex_stride_us);
    DerivedStreams {
        positions_buffer,
        normals_buffer,
        uv0_buffer,
        color_buffer,
    }
}

fn upload_skeleton_bone_buffers(
    device: &wgpu::Device,
    raw: &[u8],
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
    vc_usize: usize,
) -> Option<BoneSkinUpload> {
    let bp_raw = &raw[layout.bind_poses_start..layout.bind_poses_start + layout.bind_poses_length];
    let bind_poses_arr = extract_bind_poses(bp_raw, data.bone_count as usize)?;
    let bp_bytes: Vec<u8> = bind_poses_arr
        .iter()
        .flat_map(|m| bytemuck::bytes_of(m).iter().copied())
        .collect();
    let skinning: Vec<Mat4> = bind_poses_arr
        .iter()
        .map(Mat4::from_cols_array_2d)
        .collect();

    let bc = &raw[layout.bone_counts_start..layout.bone_counts_start + layout.bone_counts_length];
    let bw =
        &raw[layout.bone_weights_start..layout.bone_weights_start + layout.bone_weights_length];
    let (bi_buf, bw_buf) = match split_bone_weights_tail_for_gpu(bw, vc_usize) {
        Some((ib, wb)) => {
            let bi = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh {} bone_indices", data.asset_id)),
                contents: &ib,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            let bwt = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh {} bone_weights_vec4", data.asset_id)),
                contents: &wb,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            (Some(Arc::new(bi)), Some(Arc::new(bwt)))
        }
        None => {
            logger::warn!(
                "mesh {}: bone weight tail could not be repacked for GPU skinning",
                data.asset_id
            );
            (None, None)
        }
    };

    let bc_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("mesh {} bone_counts", data.asset_id)),
        contents: bc,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let bp_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("mesh {} bind_poses", data.asset_id)),
        contents: &bp_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    Some(BoneSkinUpload {
        bone_counts_buffer: Some(Arc::new(bc_buf)),
        bone_indices_buffer: bi_buf,
        bone_weights_vec4_buffer: bw_buf,
        bind_poses_buffer: Some(Arc::new(bp_buf)),
        skinning_bind_matrices: skinning,
    })
}

fn upload_synthetic_blend_bone_buffers(
    device: &wgpu::Device,
    data: &MeshUploadData,
    vc_usize: usize,
) -> BoneSkinUpload {
    let (bind_poses_arr, bone_counts, bone_weights) =
        synthetic_bone_data_for_blendshape_only(data.vertex_count);
    let bp_bytes: Vec<u8> = bind_poses_arr
        .iter()
        .flat_map(|m| bytemuck::bytes_of(m).iter().copied())
        .collect();
    let skinning: Vec<Mat4> = bind_poses_arr
        .iter()
        .map(Mat4::from_cols_array_2d)
        .collect();
    let (bi_buf, bw_buf) = split_bone_weights_tail_for_gpu(&bone_weights, vc_usize)
        .map(|(ib, wb)| {
            let bi = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh {} bone_indices synth", data.asset_id)),
                contents: &ib,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            let bwt = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh {} bone_weights_vec4 synth", data.asset_id)),
                contents: &wb,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            (Some(Arc::new(bi)), Some(Arc::new(bwt)))
        })
        .unwrap_or((None, None));
    let bc_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("mesh {} bone_counts synth", data.asset_id)),
        contents: &bone_counts,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let bp_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("mesh {} bind_poses synth", data.asset_id)),
        contents: &bp_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    BoneSkinUpload {
        bone_counts_buffer: Some(Arc::new(bc_buf)),
        bone_indices_buffer: bi_buf,
        bone_weights_vec4_buffer: bw_buf,
        bind_poses_buffer: Some(Arc::new(bp_buf)),
        skinning_bind_matrices: skinning,
    }
}

/// Bone indices/weights, bind poses, and skinning matrices for skeleton or blendshape-only paths.
///
/// Returns [`None`] when the real-skeleton bind-pose slice is invalid ([`extract_bind_poses`]).
pub(super) fn upload_bone_and_skin_buffers(
    device: &wgpu::Device,
    raw: &[u8],
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
    use_blendshapes: bool,
    vc_usize: usize,
) -> Option<BoneSkinUpload> {
    if data.bone_count > 0 {
        upload_skeleton_bone_buffers(device, raw, data, layout, vc_usize)
    } else if use_blendshapes && data.vertex_count > 0 {
        Some(upload_synthetic_blend_bone_buffers(device, data, vc_usize))
    } else {
        Some(BoneSkinUpload {
            bone_counts_buffer: None,
            bone_indices_buffer: None,
            bone_weights_vec4_buffer: None,
            bind_poses_buffer: None,
            skinning_bind_matrices: Vec::new(),
        })
    }
}

/// Packed blendshape buffer and shape count (zero when dropped or disabled).
pub(super) fn upload_blendshape_buffer(
    device: &wgpu::Device,
    gpu_limits: &GpuLimits,
    raw: &[u8],
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
    use_blendshapes: bool,
    max_buf: u64,
) -> (Option<Arc<wgpu::Buffer>>, u32) {
    if !use_blendshapes {
        return (None, 0);
    }
    match extract_blendshape_offsets(raw, layout, &data.blendshape_buffers, data.vertex_count) {
        Some((pack, n)) if !pack.is_empty() => {
            let vc_u32 = data.vertex_count.max(0) as u32;
            let n_u32 = n.max(0) as u32;
            let pack_len = pack.len() as u64;
            if pack_len > max_buf {
                logger::warn!(
                    "mesh {}: blendshapes dropped (packed size {} bytes exceeds device max_buffer_size {})",
                    data.asset_id,
                    pack_len,
                    max_buf
                );
                return (None, 0);
            }
            if plan_blendshape_bind_chunks(
                n_u32,
                vc_u32,
                gpu_limits.wgpu.max_storage_buffer_binding_size,
                gpu_limits.wgpu.min_storage_buffer_offset_alignment,
            )
            .is_none()
            {
                logger::warn!(
                    "mesh {}: blendshapes dropped ({} shapes × {} verts exceed binding limit {} or offset alignment)",
                    data.asset_id,
                    n_u32,
                    vc_u32,
                    gpu_limits.wgpu.max_storage_buffer_binding_size
                );
                return (None, 0);
            }
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh {} blendshapes", data.asset_id)),
                contents: &pack,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            (Some(Arc::new(buf)), n_u32)
        }
        _ => (None, 0),
    }
}

/// Uninitialized deform ping-pong scratch and skinning output buffers when compute paths need them.
pub(super) struct DeformOutputs {
    pub deform_temp_buffer: Option<Arc<wgpu::Buffer>>,
    pub deformed_positions_buffer: Option<Arc<wgpu::Buffer>>,
    pub deformed_normals_buffer: Option<Arc<wgpu::Buffer>>,
}

/// Allocates blend temp and/or skinning output buffers from compute flags.
pub(super) fn allocate_deform_outputs(
    device: &wgpu::Device,
    data: &MeshUploadData,
    needs_blend_compute: bool,
    needs_skin_compute: bool,
) -> DeformOutputs {
    let deform_usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST;
    let deform_temp_buffer = if needs_blend_compute {
        let len = (data.vertex_count.max(0) as u64).saturating_mul(16).max(16);
        Some(Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("mesh {} deform_temp", data.asset_id)),
            size: len,
            usage: deform_usage,
            mapped_at_creation: false,
        })))
    } else {
        None
    };

    let (deformed_positions_buffer, deformed_normals_buffer) = if needs_skin_compute {
        let len = (data.vertex_count.max(0) as u64).saturating_mul(16).max(16);
        let pos = Some(Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("mesh {} deformed_positions", data.asset_id)),
            size: len,
            usage: deform_usage,
            mapped_at_creation: false,
        })));
        let nrm = Some(Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("mesh {} deformed_normals", data.asset_id)),
            size: len,
            usage: deform_usage,
            mapped_at_creation: false,
        })));
        (pos, nrm)
    } else {
        (None, None)
    };

    DeformOutputs {
        deform_temp_buffer,
        deformed_positions_buffer,
        deformed_normals_buffer,
    }
}

pub(super) fn sum_optional_buffer_bytes(buffers: &[Option<&Arc<wgpu::Buffer>>]) -> u64 {
    buffers
        .iter()
        .filter_map(|o| o.as_ref().map(|b| b.size()))
        .sum()
}

/// Sums VRAM for all optional mesh buffers plus fixed vertex/index sizes.
pub(super) fn resident_bytes_for_mesh_upload(
    core_vb: &wgpu::Buffer,
    core_ib: &wgpu::Buffer,
    derived: &DerivedStreams,
    bone_skin: &BoneSkinUpload,
    blendshape_buffer: &Option<Arc<wgpu::Buffer>>,
    deform: &DeformOutputs,
) -> u64 {
    let mut n = core_vb.size() + core_ib.size();
    n += sum_optional_buffer_bytes(&[
        bone_skin.bone_counts_buffer.as_ref(),
        bone_skin.bone_indices_buffer.as_ref(),
        bone_skin.bone_weights_vec4_buffer.as_ref(),
        bone_skin.bind_poses_buffer.as_ref(),
        derived.positions_buffer.as_ref(),
        derived.normals_buffer.as_ref(),
        deform.deform_temp_buffer.as_ref(),
        deform.deformed_positions_buffer.as_ref(),
        deform.deformed_normals_buffer.as_ref(),
        derived.uv0_buffer.as_ref(),
        derived.color_buffer.as_ref(),
    ]);
    if let Some(ref b) = blendshape_buffer {
        n += b.size();
    }
    n
}
