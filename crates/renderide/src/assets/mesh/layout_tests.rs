//! Unit tests for [`super::layout`] (host mesh buffer layout and stream extraction).

use glam::Mat4;

use super::layout::{
    color_float4_stream_bytes, compute_index_count, compute_mesh_buffer_layout,
    compute_vertex_stride, extract_blendshape_offsets,
    extract_float3_position_normal_as_vec4_streams, index_bytes_per_element,
    split_bone_weights_tail_for_gpu, uv0_float2_stream_bytes, vertex_float2_stream_bytes,
    vertex_float4_stream_bytes, BLENDSHAPE_SPARSE_ENTRY_SIZE,
};
use crate::shared::{
    BlendshapeBufferDescriptor, BlendshapeDataFlags, IndexBufferFormat, SubmeshBufferDescriptor,
    SubmeshTopology, VertexAttributeDescriptor, VertexAttributeFormat, VertexAttributeType,
};

#[test]
fn layout_no_bones_no_blend_matches_stride() {
    let sub = vec![SubmeshBufferDescriptor {
        topology: SubmeshTopology::default(),
        index_start: 0,
        index_count: 3,
        bounds: crate::shared::RenderBoundingBox::default(),
    }];
    let ic = compute_index_count(&sub);
    let l = compute_mesh_buffer_layout(32, 2, ic, 2, 0, 0, None).unwrap();
    assert_eq!(l.vertex_size, 64);
    assert_eq!(l.index_buffer_start, 64);
    assert_eq!(l.index_buffer_length, 6);
    assert_eq!(l.bone_counts_length, 2);
    assert_eq!(l.bone_weights_length, 0);
    assert_eq!(l.bind_poses_length, 0);
    assert_eq!(l.total_buffer_length, 64 + 6 + 2);
}

#[test]
fn layout_negative_bone_counts_clamped() {
    let sub = vec![SubmeshBufferDescriptor {
        topology: SubmeshTopology::default(),
        index_start: 0,
        index_count: 3,
        bounds: crate::shared::RenderBoundingBox::default(),
    }];
    let ic = compute_index_count(&sub);
    let l = compute_mesh_buffer_layout(32, 2, ic, 2, -1, -1, None).unwrap();
    assert_eq!(l.vertex_size, 64);
    assert_eq!(l.index_buffer_start, 64);
    assert_eq!(l.index_buffer_length, 6);
    assert_eq!(l.bone_counts_length, 2);
    assert_eq!(l.bone_weights_length, 0);
    assert_eq!(l.bind_poses_length, 0);
    assert_eq!(l.total_buffer_length, 64 + 6 + 2);
}

#[test]
fn vertex_stride_sum() {
    let attrs = [
        VertexAttributeDescriptor {
            attribute: VertexAttributeType::Position,
            format: VertexAttributeFormat::Float32,
            dimensions: 3,
        },
        VertexAttributeDescriptor {
            attribute: VertexAttributeType::Normal,
            format: VertexAttributeFormat::Float32,
            dimensions: 3,
        },
    ];
    assert_eq!(compute_vertex_stride(&attrs), 24);
}

#[test]
fn position_stream_synthesizes_normals_when_normal_missing() {
    let attrs = [VertexAttributeDescriptor {
        attribute: VertexAttributeType::Position,
        format: VertexAttributeFormat::Float32,
        dimensions: 3,
    }];
    let mut raw = Vec::new();
    for value in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
        raw.extend_from_slice(&value.to_le_bytes());
    }

    let (pos, nrm) =
        extract_float3_position_normal_as_vec4_streams(&raw, 2, 12, &attrs).expect("streams");
    let pos0: [f32; 4] = bytemuck::pod_read_unaligned(&pos[..16]);
    let pos1: [f32; 4] = bytemuck::pod_read_unaligned(&pos[16..32]);
    let nrm0: [f32; 4] = bytemuck::pod_read_unaligned(&nrm[..16]);
    let nrm1: [f32; 4] = bytemuck::pod_read_unaligned(&nrm[16..32]);

    assert_eq!(pos0, [1.0, 2.0, 3.0, 1.0]);
    assert_eq!(pos1, [4.0, 5.0, 6.0, 1.0]);
    assert_eq!(nrm0, [0.0, 0.0, 1.0, 0.0]);
    assert_eq!(nrm1, [0.0, 0.0, 1.0, 0.0]);
}

#[test]
fn split_bone_weights_four_influences_roundtrip() {
    let mut tail = Vec::new();
    for v in 0..2u8 {
        for k in 0..4u8 {
            let w = (v + k) as f32 * 0.1;
            let j = (k as i32) + (v as i32) * 10;
            tail.extend_from_slice(&w.to_le_bytes());
            tail.extend_from_slice(&j.to_le_bytes());
        }
    }
    let (idx, wt) = split_bone_weights_tail_for_gpu(&tail, 2).expect("split");
    let w0 = f32::from_le_bytes(wt[0..4].try_into().unwrap());
    let i0 = u32::from_le_bytes(idx[0..4].try_into().unwrap());
    assert!((w0 - 0.0).abs() < 1e-5);
    assert_eq!(i0, 0);

    // Vertex 1, first influence (k=0): w=0.1, j=10
    let w1_0 = f32::from_le_bytes(wt[16..20].try_into().unwrap());
    let i1_0 = u32::from_le_bytes(idx[16..20].try_into().unwrap());
    assert!((w1_0 - 0.1).abs() < 1e-5);
    assert_eq!(i1_0, 10);
}

#[test]
fn split_bone_weights_negative_index_zeroes_weight() {
    let mut tail = Vec::new();
    tail.extend_from_slice(&0.5f32.to_le_bytes());
    tail.extend_from_slice(&(-1i32).to_le_bytes());
    tail.extend_from_slice(&0f32.to_le_bytes());
    tail.extend_from_slice(&0i32.to_le_bytes());
    tail.extend_from_slice(&0f32.to_le_bytes());
    tail.extend_from_slice(&0i32.to_le_bytes());
    tail.extend_from_slice(&0f32.to_le_bytes());
    tail.extend_from_slice(&0i32.to_le_bytes());
    let (idx, wt) = split_bone_weights_tail_for_gpu(&tail, 1).expect("split");
    let w0 = f32::from_le_bytes(wt[0..4].try_into().unwrap());
    let i0 = u32::from_le_bytes(idx[0..4].try_into().unwrap());
    assert!((w0 - 0.0).abs() < 1e-5);
    assert_eq!(i0, 0u32);
}

/// Unity uploads inverse bind matrices; the renderer stores them as [`Mat4::from_cols_array_2d`]
/// without an extra `.inverse()` (see [`crate::assets::mesh::GpuMesh::skinning_bind_matrices`]).
#[test]
fn unity_bindpose_raw_matches_glam_columns_not_inverse() {
    let expected = Mat4::from_translation(glam::Vec3::new(1.0, 2.0, 3.0));
    let a = expected.to_cols_array();
    let raw: [[f32; 4]; 4] = [
        [a[0], a[1], a[2], a[3]],
        [a[4], a[5], a[6], a[7]],
        [a[8], a[9], a[10], a[11]],
        [a[12], a[13], a[14], a[15]],
    ];
    let stored = Mat4::from_cols_array_2d(&raw);
    assert!(stored.abs_diff_eq(expected, 1e-5));
    assert!(!stored.abs_diff_eq(expected.inverse(), 1e-2));
}

#[test]
fn uv0_float2_zeros_when_missing() {
    let attrs = [
        VertexAttributeDescriptor {
            attribute: VertexAttributeType::Position,
            format: VertexAttributeFormat::Float32,
            dimensions: 3,
        },
        VertexAttributeDescriptor {
            attribute: VertexAttributeType::Normal,
            format: VertexAttributeFormat::Float32,
            dimensions: 3,
        },
    ];
    let stride = 24usize;
    let verts = 2usize;
    let raw = vec![0u8; verts * stride];
    let out = uv0_float2_stream_bytes(&raw, verts, stride, &attrs).expect("uv stream");
    assert_eq!(out.len(), verts * 8);
    assert!(out.iter().all(|&b| b == 0));
}

#[test]
fn vertex_float2_extracts_uv1_stream() {
    let attrs = [
        VertexAttributeDescriptor {
            attribute: VertexAttributeType::Position,
            format: VertexAttributeFormat::Float32,
            dimensions: 3,
        },
        VertexAttributeDescriptor {
            attribute: VertexAttributeType::UV1,
            format: VertexAttributeFormat::Float32,
            dimensions: 2,
        },
    ];
    let mut raw = Vec::new();
    raw.extend_from_slice(&0.0f32.to_le_bytes());
    raw.extend_from_slice(&0.0f32.to_le_bytes());
    raw.extend_from_slice(&0.0f32.to_le_bytes());
    raw.extend_from_slice(&1.25f32.to_le_bytes());
    raw.extend_from_slice(&2.5f32.to_le_bytes());

    let out = vertex_float2_stream_bytes(&raw, 1, 20, &attrs, VertexAttributeType::UV1)
        .expect("uv1 stream");
    let uv: [f32; 2] = bytemuck::pod_read_unaligned(&out[..8]);
    assert_eq!(uv, [1.25, 2.5]);
}

#[test]
fn vertex_float4_defaults_when_tangent_missing() {
    let attrs = [VertexAttributeDescriptor {
        attribute: VertexAttributeType::Position,
        format: VertexAttributeFormat::Float32,
        dimensions: 3,
    }];
    let raw = vec![0u8; 12];
    let out = vertex_float4_stream_bytes(
        &raw,
        1,
        12,
        &attrs,
        VertexAttributeType::Tangent,
        [1.0, 0.0, 0.0, 1.0],
    )
    .expect("tangent stream");
    let tangent: [f32; 4] = bytemuck::pod_read_unaligned(&out[..16]);
    assert_eq!(tangent, [1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn color_stream_defaults_to_opaque_white_when_missing() {
    let attrs = [VertexAttributeDescriptor {
        attribute: VertexAttributeType::Position,
        format: VertexAttributeFormat::Float32,
        dimensions: 3,
    }];
    let raw = vec![0u8; 12];
    let out = color_float4_stream_bytes(&raw, 1, 12, &attrs).expect("color stream");
    let rgba: [f32; 4] = bytemuck::pod_read_unaligned(&out[..16]);
    assert_eq!(rgba, [1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn color_stream_decodes_unorm8_rgba() {
    let attrs = [VertexAttributeDescriptor {
        attribute: VertexAttributeType::Color,
        format: VertexAttributeFormat::UNorm8,
        dimensions: 4,
    }];
    let raw = vec![255u8, 128u8, 0u8, 64u8];
    let out = color_float4_stream_bytes(&raw, 1, 4, &attrs).expect("color stream");
    let rgba: [f32; 4] = bytemuck::pod_read_unaligned(&out[..16]);
    assert!((rgba[0] - 1.0).abs() < 1e-6);
    assert!((rgba[1] - (128.0 / 255.0)).abs() < 1e-6);
    assert!((rgba[2] - 0.0).abs() < 1e-6);
    assert!((rgba[3] - (64.0 / 255.0)).abs() < 1e-6);
}

#[test]
fn color_stream_decodes_uint8_rgba_as_normalized_color() {
    let attrs = [VertexAttributeDescriptor {
        attribute: VertexAttributeType::Color,
        format: VertexAttributeFormat::UInt8,
        dimensions: 4,
    }];
    let raw = vec![0u8, 64u8, 128u8, 255u8];
    let out = color_float4_stream_bytes(&raw, 1, 4, &attrs).expect("color stream");
    let rgba: [f32; 4] = bytemuck::pod_read_unaligned(&out[..16]);
    assert!((rgba[0] - 0.0).abs() < 1e-6);
    assert!((rgba[1] - (64.0 / 255.0)).abs() < 1e-6);
    assert!((rgba[2] - (128.0 / 255.0)).abs() < 1e-6);
    assert!((rgba[3] - 1.0).abs() < 1e-6);
}

#[test]
fn extract_blendshape_sparse_keeps_nonzero_position_rows_only() {
    let vertex_count = 2i32;
    let attrs = [
        VertexAttributeDescriptor {
            attribute: VertexAttributeType::Position,
            format: VertexAttributeFormat::Float32,
            dimensions: 3,
        },
        VertexAttributeDescriptor {
            attribute: VertexAttributeType::Normal,
            format: VertexAttributeFormat::Float32,
            dimensions: 3,
        },
    ];
    let stride = compute_vertex_stride(&attrs);
    let sub = [SubmeshBufferDescriptor {
        topology: SubmeshTopology::default(),
        index_start: 0,
        index_count: 3,
        bounds: crate::shared::RenderBoundingBox::default(),
    }];
    let ic = compute_index_count(&sub);
    let blend = [BlendshapeBufferDescriptor {
        blendshape_index: 0,
        frame_index: 0,
        frame_weight: 1.0,
        data_flags: BlendshapeDataFlags(BlendshapeDataFlags::POSITIONS),
    }];
    let layout = compute_mesh_buffer_layout(
        stride,
        vertex_count,
        ic,
        index_bytes_per_element(IndexBufferFormat::UInt16),
        0,
        0,
        Some(&blend),
    )
    .expect("layout");
    let mut full = vec![0u8; layout.total_buffer_length];
    let off = layout.blendshape_data_start;
    let ax = 1.0f32.to_le_bytes();
    full[off..off + 4].copy_from_slice(&ax);
    let pack = extract_blendshape_offsets(&full, &layout, &blend, vertex_count).expect("pack");
    assert_eq!(pack.num_blendshapes, 1);
    assert_eq!(pack.shape_ranges[0], (0, 1));
    assert_eq!(pack.sparse_deltas.len(), BLENDSHAPE_SPARSE_ENTRY_SIZE);
}
