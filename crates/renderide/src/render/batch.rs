//! Per-space draw batch for rendering.
//!
//! Extension point for batch structure, draw ordering.

use nalgebra::Matrix4;

use crate::gpu::PipelineVariant;
use crate::shared::RenderTransform;

/// Single draw within a batch. Sorted by (pipeline_variant, material_id, mesh_asset_id).
#[derive(Clone)]
pub struct DrawEntry {
    /// Model-to-world matrix.
    pub model_matrix: Matrix4<f32>,
    /// Mesh asset handle.
    pub mesh_asset_id: i32,
    /// Whether this draw uses skinned mesh pipeline.
    pub is_skinned: bool,
    /// Material handle (for future material pipeline).
    pub material_id: i32,
    /// Bone transform node IDs for skinned meshes.
    pub bone_transform_ids: Option<Vec<i32>>,
    /// Pipeline variant derived from is_skinned, use_debug_uv, and mesh has_uvs.
    pub pipeline_variant: PipelineVariant,
}

/// Per-space draw batch for rendering.
#[derive(Clone)]
pub struct SpaceDrawBatch {
    /// Scene/space identifier.
    pub space_id: i32,
    /// Whether this is an overlay.
    pub is_overlay: bool,
    /// View transform for this space.
    pub view_transform: RenderTransform,
    /// Draws sorted by (pipeline_variant, material_id, mesh_asset_id).
    pub draws: Vec<DrawEntry>,
}
