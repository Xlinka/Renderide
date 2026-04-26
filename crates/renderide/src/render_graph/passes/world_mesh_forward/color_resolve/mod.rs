//! HDR-aware MSAA color resolve for the world-mesh forward path.
//!
//! Replaces wgpu's automatic linear `resolve_target` average. A linear average of HDR samples is
//! perceptually wrong at stark contrast edges: a pixel partially covered by a very bright sample
//! (specular spark, emissive, sun) and partially by a dark sample averages to a value that, after
//! tonemapping, looks too bright (or too dark in the inverse case), producing visibly aliased
//! silhouettes between bright and dark surfaces even with high MSAA.
//!
//! This pass implements the Karis bracket used by Filament's `customResolveAsSubpass`: each
//! sample is compressed by `x / (1 + max3(x))`, the compressed values are linearly averaged, and
//! the result is decompressed by `y / (1 - max3(y))`. The compress / average / uncompress
//! sandwich approximates "tonemap each sample, average, untonemap" while keeping an HDR result
//! for downstream bloom and tonemap to consume.
//!
//! Only registered in the graph when MSAA is active (sample count > 1). When MSAA is off, the
//! intersect pass writes directly to the single-sample `scene_color_hdr` and no resolve pass is
//! needed.

mod pipeline;

use std::num::NonZeroU32;
use std::sync::OnceLock;

use pipeline::{MsaaResolveHdrPipelineCache, ResolveParamsUbo};

use crate::render_graph::compiled::RenderPassTemplate;
use crate::render_graph::context::RasterPassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::pass::{PassBuilder, RasterPass};
use crate::render_graph::resources::{ImportedTextureHandle, TextureAccess, TextureHandle};

/// Graph handles for [`WorldMeshForwardColorResolvePass`].
#[derive(Clone, Copy, Debug)]
pub struct WorldMeshForwardColorResolveGraphResources {
    /// Multisampled HDR scene-color source produced by the forward opaque + intersect passes.
    pub scene_color_hdr_msaa: TextureHandle,
    /// Single-sample HDR destination consumed by post-processing and scene compose.
    pub scene_color_hdr: TextureHandle,
}

/// Resolves multisampled HDR scene color to single-sample HDR using the Karis bracket.
pub struct WorldMeshForwardColorResolvePass {
    resources: WorldMeshForwardColorResolveGraphResources,
    pipelines: &'static MsaaResolveHdrPipelineCache,
}

impl WorldMeshForwardColorResolvePass {
    /// Creates a color-resolve pass instance.
    pub fn new(resources: WorldMeshForwardColorResolveGraphResources) -> Self {
        Self {
            resources,
            pipelines: pipeline_cache(),
        }
    }
}

fn pipeline_cache() -> &'static MsaaResolveHdrPipelineCache {
    static CACHE: OnceLock<MsaaResolveHdrPipelineCache> = OnceLock::new();
    CACHE.get_or_init(MsaaResolveHdrPipelineCache::default)
}

impl RasterPass for WorldMeshForwardColorResolvePass {
    fn name(&self) -> &str {
        "WorldMeshForwardColorResolve"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.read_texture_resource(
            self.resources.scene_color_hdr_msaa,
            TextureAccess::Sampled {
                stages: wgpu::ShaderStages::FRAGMENT,
            },
        );
        {
            let mut r = b.raster();
            // Clear is fine: the fullscreen draw fills every pixel of `scene_color_hdr` with the
            // resolved value, so we don't depend on prior contents.
            r.color(
                self.resources.scene_color_hdr,
                wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                Option::<ImportedTextureHandle>::None,
            );
        }
        Ok(())
    }

    fn multiview_mask_override(
        &self,
        ctx: &RasterPassCtx<'_, '_>,
        template: &RenderPassTemplate,
    ) -> Option<NonZeroU32> {
        let stereo = ctx
            .frame
            .as_ref()
            .is_some_and(|frame| frame.view.multiview_stereo);
        if stereo {
            NonZeroU32::new(3)
        } else {
            template.multiview_mask
        }
    }

    fn record(
        &self,
        ctx: &mut RasterPassCtx<'_, '_>,
        rpass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderPassError> {
        profiling::scope!("world_mesh_forward::color_resolve_record");
        let Some(frame) = ctx.frame.as_ref() else {
            return Err(RenderPassError::MissingFrameParams {
                pass: self.name().to_string(),
            });
        };
        let Some(graph_resources) = ctx.graph_resources else {
            return Err(RenderPassError::MissingFrameParams {
                pass: self.name().to_string(),
            });
        };
        let Some(src) = graph_resources.transient_texture(self.resources.scene_color_hdr_msaa)
        else {
            return Err(RenderPassError::MissingFrameParams {
                pass: format!(
                    "{} (missing transient scene_color_hdr_msaa {:?})",
                    self.name(),
                    self.resources.scene_color_hdr_msaa
                ),
            });
        };

        // Skip when the runtime sample count is 1: there are no multisamples to resolve, and
        // the intersect pass has already written directly to `scene_color_hdr`. The pass is
        // expected to be omitted from the graph in that case (the build site only inserts the
        // resolve when MSAA is active), but guard at runtime to be safe.
        let sample_count = frame.view.sample_count;
        if sample_count <= 1 {
            return Ok(());
        }

        let multiview_stereo = frame.view.multiview_stereo;
        let pipeline = self
            .pipelines
            .pipeline(ctx.device, src.texture.format(), multiview_stereo);

        // Upload sample count to the per-frame UBO. Single u32 plus 12 bytes of padding.
        let params = ResolveParamsUbo {
            sample_count,
            _pad: [0; 3],
        };
        let params_ubo = self.pipelines.params_ubo(ctx.device);
        ctx.queue
            .write_buffer(params_ubo, 0, bytemuck::bytes_of(&params));

        // Build per-frame bind group(s) over the multisampled source. WGSL has no
        // `texture_multisampled_2d_array` in naga 29, so the stereo path binds two single-layer
        // views; the shader picks between them with `@builtin(view_index)`.
        let bind_group = if multiview_stereo {
            let layer0 = src.texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("msaa_resolve_hdr_src_msaa_left"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: 0,
                array_layer_count: Some(1),
                ..Default::default()
            });
            let layer1 = src.texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("msaa_resolve_hdr_src_msaa_right"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: 1,
                array_layer_count: Some(1),
                ..Default::default()
            });
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("msaa_resolve_hdr_bg_multiview"),
                layout: self.pipelines.bind_group_layout_multiview(ctx.device),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_ubo.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&layer0),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&layer1),
                    },
                ],
            })
        } else {
            let view = src.texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("msaa_resolve_hdr_src_msaa"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: 0,
                array_layer_count: Some(1),
                ..Default::default()
            });
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("msaa_resolve_hdr_bg_mono"),
                layout: self.pipelines.bind_group_layout_mono(ctx.device),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_ubo.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                ],
            })
        };

        rpass.set_pipeline(pipeline.as_ref());
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
        Ok(())
    }
}
