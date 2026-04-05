//! GPU device, adapter, and swapchain configuration.

mod blendshape_bind_chunks;
mod context;
pub mod mesh_preprocess;

pub use blendshape_bind_chunks::plan_blendshape_bind_chunks;
pub use context::GpuContext;
pub use mesh_preprocess::MeshPreprocessPipelines;
