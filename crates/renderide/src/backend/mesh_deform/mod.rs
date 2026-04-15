//! Mesh skinning / blendshape scatter compute preprocess, sparse buffer checks, and per-draw
//! uniform packing for `@group(2)` in the world mesh forward pass.

mod blendshape_bind_chunks;
mod mesh_preprocess;
mod per_draw_uniforms;
mod scratch;

pub use blendshape_bind_chunks::{
    blendshape_sparse_buffers_fit_device, plan_blendshape_scatter_chunks,
    BLENDSHAPE_SPARSE_MIN_BUFFER_BYTES,
};
pub use mesh_preprocess::MeshPreprocessPipelines;
pub use per_draw_uniforms::{
    write_per_draw_uniform_slab, PaddedPerDrawUniforms, WgslMat3x3, INITIAL_PER_DRAW_UNIFORM_SLOTS,
    PER_DRAW_UNIFORM_STRIDE,
};
pub use scratch::{advance_slab_cursor, MeshDeformScratch};
