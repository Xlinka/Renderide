//! Cached pipelines and bind layout for [`super::SceneColorComposePass`].
//!
//! WGSL is sourced from the build-time embedded shader registry
//! ([`crate::embedded_shaders::embedded_target_wgsl`]): the single source
//! `shaders/source/post/scene_color_compose.wgsl` is composed twice with `#ifdef MULTIVIEW` into
//! `scene_color_compose_default` and `scene_color_compose_multiview` by the build script.

use std::sync::Arc;

use crate::embedded_shaders::{
    SCENE_COLOR_COMPOSE_DEFAULT_WGSL, SCENE_COLOR_COMPOSE_MULTIVIEW_WGSL,
};
use crate::render_graph::gpu_cache::{
    create_d2_array_view, create_fullscreen_render_pipeline, create_linear_clamp_sampler,
    create_wgsl_shader_module, BindGroupMap, FullscreenRenderPipelineDesc, OnceGpu,
    RenderPipelineMap,
};

/// Debug label for the mono variant pipeline.
const PIPELINE_LABEL_MONO: &str = "scene_color_compose_default";
/// Debug label for the multiview variant pipeline.
const PIPELINE_LABEL_MULTIVIEW: &str = "scene_color_compose_multiview";

/// GPU state shared by all compose passes (bind layout + sampler).
pub(super) struct SceneColorComposePipelineCache {
    /// Bind group layout shared by mono and multiview variants.
    bind_group_layout: OnceGpu<wgpu::BindGroupLayout>,
    /// Linear sampler shared by all compose draws.
    sampler: OnceGpu<wgpu::Sampler>,
    /// Mono pipelines keyed by output color format.
    mono: RenderPipelineMap<wgpu::TextureFormat>,
    /// Multiview pipelines keyed by output color format.
    multiview: RenderPipelineMap<wgpu::TextureFormat>,
    /// Bind groups keyed by scene-color texture identity + multiview flag; avoids rebuilding
    /// on every frame when the transient pool reuses the same allocation.
    bind_groups: BindGroupMap<(wgpu::Texture, bool)>,
}

impl Default for SceneColorComposePipelineCache {
    fn default() -> Self {
        Self {
            bind_group_layout: OnceGpu::default(),
            sampler: OnceGpu::default(),
            mono: RenderPipelineMap::default(),
            multiview: RenderPipelineMap::default(),
            bind_groups: BindGroupMap::with_max_entries(MAX_CACHED_BIND_GROUPS),
        }
    }
}

impl SceneColorComposePipelineCache {
    /// Linear clamp sampler for HDR scene color.
    pub(super) fn sampler(&self, device: &wgpu::Device) -> &wgpu::Sampler {
        self.sampler
            .get_or_create(|| create_linear_clamp_sampler(device, "scene_color_compose"))
    }

    fn bind_group_layout(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.bind_group_layout.get_or_create(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scene_color_compose"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            })
        })
    }

    /// Returns or builds a render pipeline for `output_format` and multiview stereo.
    pub(super) fn pipeline(
        &self,
        device: &wgpu::Device,
        output_format: wgpu::TextureFormat,
        multiview_stereo: bool,
    ) -> Arc<wgpu::RenderPipeline> {
        let map = if multiview_stereo {
            &self.multiview
        } else {
            &self.mono
        };
        map.get_or_create(output_format, |output_format| {
            logger::debug!(
                "scene_color_compose: building pipeline (dst format = {:?}, multiview = {})",
                output_format,
                multiview_stereo
            );
            let (label, source) = if multiview_stereo {
                (PIPELINE_LABEL_MULTIVIEW, SCENE_COLOR_COMPOSE_MULTIVIEW_WGSL)
            } else {
                (PIPELINE_LABEL_MONO, SCENE_COLOR_COMPOSE_DEFAULT_WGSL)
            };
            let shader = create_wgsl_shader_module(device, label, source);
            let bind_group_layout = self.bind_group_layout(device);
            create_fullscreen_render_pipeline(
                device,
                FullscreenRenderPipelineDesc {
                    label,
                    bind_group_layouts: &[Some(bind_group_layout)],
                    shader: &shader,
                    fragment_entry: "fs_main",
                    output_format: *output_format,
                    blend: None,
                    multiview_stereo,
                },
            )
        })
    }

    /// Bind group for one frame's scene-color texture, cached by `(Texture, multiview_stereo)`.
    pub(super) fn bind_group(
        &self,
        device: &wgpu::Device,
        scene_color_texture: &wgpu::Texture,
        multiview_stereo: bool,
    ) -> wgpu::BindGroup {
        let key = (scene_color_texture.clone(), multiview_stereo);
        self.bind_groups.get_or_create(key, |key| {
            let (scene_color_texture, multiview_stereo) = key;
            let view = create_d2_array_view(
                scene_color_texture,
                "scene_color_compose_sampled",
                *multiview_stereo,
            );
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scene_color_compose"),
                layout: self.bind_group_layout(device),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(self.sampler(device)),
                    },
                ],
            })
        })
    }
}

/// Upper bound for cached scene-color-compose bind groups before the cache is flushed.
///
/// Normally one or two entries (mono + multiview). The cap protects against unbounded growth
/// when the transient pool cycles allocations (resize / MSAA toggle).
const MAX_CACHED_BIND_GROUPS: usize = 8;
