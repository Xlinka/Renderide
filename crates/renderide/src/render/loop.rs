//! Render loop: executes one frame via the render graph.
//!
//! Extension point for RenderGraph passes (mirrors, post, UI, probes).

use super::pass::{reverse_z_projection, MeshRenderPass, RenderGraph, RenderGraphContext};
use super::SpaceDrawBatch;
use crate::gpu::{GpuState, PipelineManager};
use crate::session::Session;

/// Encapsulates the render frame logic.
pub struct RenderLoop {
    pipeline_manager: PipelineManager,
    graph: RenderGraph,
}

impl RenderLoop {
    /// Creates a new render loop with pipelines for the given device and config.
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let mut graph = RenderGraph::new();
        graph.add_pass(Box::new(MeshRenderPass::new()));
        Self {
            pipeline_manager: PipelineManager::new(device, config),
            graph,
        }
    }

    /// Renders one frame: clear, draw batches. Caller must present the returned texture.
    pub fn render_frame(
        &mut self,
        gpu: &mut GpuState,
        session: &Session,
        draw_batches: &[SpaceDrawBatch],
    ) -> Result<wgpu::SurfaceTexture, wgpu::SurfaceError> {
        let output = gpu.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = gpu
            .depth_texture
            .as_ref()
            .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()));

        let aspect = gpu.config.width as f32 / gpu.config.height.max(1) as f32;
        let proj = reverse_z_projection(
            aspect,
            session.desktop_fov().to_radians(),
            session.near_clip().max(0.01),
            session.far_clip(),
        );

        let viewport = (gpu.config.width, gpu.config.height);
        let mut ctx = RenderGraphContext {
            gpu,
            session,
            draw_batches,
            pipeline_manager: &mut self.pipeline_manager,
            viewport,
            color_view: &view,
            depth_view: depth_view.as_ref(),
            proj,
        };

        self.graph.execute(&mut ctx).map_err(|e| match e {
            super::pass::RenderPassError::Surface(s) => s,
        })?;

        Ok(output)
    }
}
