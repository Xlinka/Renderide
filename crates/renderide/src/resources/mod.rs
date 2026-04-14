//! GPU resource pools and VRAM hooks (meshes and Texture2D).

mod budget;
mod mesh_pool;
mod render_texture_pool;
mod texture_pool;

pub use budget::{
    MeshResidencyMeta, NoopStreamingPolicy, ResidencyTier, StreamingPolicy, TextureResidencyMeta,
    VramAccounting, VramResourceKind,
};
pub use mesh_pool::MeshPool;
pub use render_texture_pool::{GpuRenderTexture, RenderTexturePool};
pub use texture_pool::{GpuTexture2d, Texture2dSamplerState, TexturePool};

/// Common surface for resident GPU resources (extend for textures, buffers, etc.).
pub trait GpuResource {
    /// Approximate GPU memory for accounting.
    fn resident_bytes(&self) -> u64;
    /// Host asset id.
    fn asset_id(&self) -> i32;
}
