//! Dedicated skybox/background rendering for the world-mesh forward opaque pass.
//!
//! Procedural skyboxes are driven by [`crate::scene::RenderSpaceState::skybox_material_asset_id`]
//! plus the normal material/shader IPC streams. The renderer resolves the skybox material
//! directly here rather than trying to batch it through the mesh material path.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use parking_lot::Mutex;
use wgpu::util::DeviceExt;

use crate::assets::material::{MaterialPropertyStore, MaterialPropertyValue, PropertyIdRegistry};
use crate::assets::util::normalize_unity_shader_lookup_key;
use crate::embedded_shaders::embedded_target_wgsl;
use crate::render_graph::camera::{
    clamp_desktop_fov_degrees, effective_head_output_clip_planes, reverse_z_perspective,
    view_matrix_for_world_mesh_render_space,
};
use crate::render_graph::frame_params::{FrameRenderParams, PreparedWorldMeshForwardFrame};

/// Embedded shader stem for mono procedural skybox rendering.
const WGSL_STEM_MONO: &str = "procedural_skybox_default";
/// Embedded shader stem for stereo multiview procedural skybox rendering.
const WGSL_STEM_MULTIVIEW: &str = "procedural_skybox_multiview";
/// Normalized Unity shader names observed for the stock procedural skybox shader.
///
/// Resonite currently reports `proceduralskybox` in the shader-routes HUD, while the original
/// Unity shader name is `ProceduralSky`.
const PROCEDURAL_SKY_UNITY_KEYS: &[&str] = &["proceduralsky", "proceduralskybox"];

/// CPU uniform payload for [`WGSL_STEM_MONO`] / [`WGSL_STEM_MULTIVIEW`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ProceduralSkyUniforms {
    view_to_world_left: [[f32; 4]; 4],
    view_to_world_right: [[f32; 4]; 4],
    inv_proj_left: [[f32; 4]; 4],
    inv_proj_right: [[f32; 4]; 4],
    sun_direction: [f32; 4],
    sun_color: [f32; 4],
    sky_tint: [f32; 4],
    ground_color: [f32; 4],
    params0: [f32; 4],
}

#[derive(Clone, Copy, Debug)]
struct ProceduralSkyViewMatrices {
    view_to_world_left: Mat4,
    view_to_world_right: Mat4,
    inv_proj_left: Mat4,
    inv_proj_right: Mat4,
}

#[derive(Clone, Copy, Debug, Default)]
struct ProceduralSkyPropertyIds {
    sun_disk: Option<i32>,
    sun_size: Option<i32>,
    sun_direction: Option<i32>,
    sun_color: Option<i32>,
    atmosphere_thickness: Option<i32>,
    sky_tint: Option<i32>,
    ground_color: Option<i32>,
    exposure: Option<i32>,
}

impl ProceduralSkyPropertyIds {
    fn from_registry(registry: &PropertyIdRegistry) -> Self {
        Self {
            sun_disk: registry.lookup("_SunDisk"),
            sun_size: registry.lookup("_SunSize"),
            sun_direction: registry.lookup("_SunDirection"),
            sun_color: registry.lookup("_SunColor"),
            atmosphere_thickness: registry.lookup("_AtmosphereThickness"),
            sky_tint: registry.lookup("_SkyTint"),
            ground_color: registry.lookup("_GroundColor"),
            exposure: registry.lookup("_Exposure"),
        }
    }
}

/// GPU state shared across every procedural skybox draw.
pub(super) struct ProceduralSkyPipelineCache {
    bind_group_layout: OnceLock<wgpu::BindGroupLayout>,
    mono: Mutex<HashMap<(wgpu::TextureFormat, wgpu::TextureFormat, u32), Arc<wgpu::RenderPipeline>>>,
    multiview:
        Mutex<HashMap<(wgpu::TextureFormat, wgpu::TextureFormat, u32), Arc<wgpu::RenderPipeline>>>,
}

impl Default for ProceduralSkyPipelineCache {
    fn default() -> Self {
        Self {
            bind_group_layout: OnceLock::new(),
            mono: Mutex::new(HashMap::new()),
            multiview: Mutex::new(HashMap::new()),
        }
    }
}

impl ProceduralSkyPipelineCache {
    fn bind_group_layout(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.bind_group_layout.get_or_init(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("procedural_skybox"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            })
        })
    }

    fn pipeline(
        &self,
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        sample_count: u32,
        multiview: bool,
    ) -> Arc<wgpu::RenderPipeline> {
        let key = (color_format, depth_format, sample_count.max(1));
        let cache = if multiview { &self.multiview } else { &self.mono };
        let mut guard = cache.lock();
        if let Some(existing) = guard.get(&key) {
            return Arc::clone(existing);
        }

        let stem = if multiview {
            WGSL_STEM_MULTIVIEW
        } else {
            WGSL_STEM_MONO
        };
        #[expect(
            clippy::expect_used,
            reason = "embedded procedural skybox shader is required; absence is a build regression"
        )]
        let source = embedded_target_wgsl(stem)
            .expect("procedural_skybox: embedded shader missing (build script regression)");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(stem),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(stem),
            bind_group_layouts: &[Some(self.bind_group_layout(device))],
            immediate_size: 0,
        });
        let pipeline = Arc::new(
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(stem),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: sample_count.max(1),
                    ..Default::default()
                },
                multiview_mask: multiview
                    .then(|| std::num::NonZeroU32::new(3))
                    .flatten(),
                cache: None,
            }),
        );
        guard.insert(key, Arc::clone(&pipeline));
        pipeline
    }

    fn bind_group(
        &self,
        device: &wgpu::Device,
        uniforms: &ProceduralSkyUniforms,
    ) -> (wgpu::Buffer, wgpu::BindGroup) {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("procedural_skybox_uniforms"),
            contents: bytemuck::bytes_of(uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("procedural_skybox"),
            layout: self.bind_group_layout(device),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        (buffer, bind_group)
    }
}

pub(super) fn record_procedural_skybox(
    rpass: &mut wgpu::RenderPass<'_>,
    device: &wgpu::Device,
    frame: &FrameRenderParams<'_>,
    prepared: &PreparedWorldMeshForwardFrame,
) {
    let Some(render_space_id) = frame.view.render_space_id else {
        return;
    };
    let Some(space) = frame.shared.scene.space(render_space_id) else {
        return;
    };
    if space.is_overlay || space.skybox_material_asset_id < 0 {
        return;
    }

    let material_id = space.skybox_material_asset_id;
    let materials = frame.shared.materials;
    let store = materials.material_property_store();
    let Some(shader_asset_id) = store.shader_asset_for_material(material_id) else {
        return;
    };
    let Some(router) = materials.material_registry().map(|registry| &registry.router) else {
        return;
    };
    let Some(display_name) = router.display_name_for_shader_asset(shader_asset_id) else {
        return;
    };
    if !is_procedural_sky_shader_name(display_name) {
        return;
    }

    let Some(uniforms) = ProceduralSkyUniforms::from_frame(
        frame,
        render_space_id,
        material_id,
        prepared.pipeline.use_multiview,
    ) else {
        return;
    };

    let pipeline = procedural_skybox_pipelines().pipeline(
        device,
        frame.view.scene_color_format,
        frame.view.depth_texture.format(),
        frame.view.sample_count,
        prepared.pipeline.use_multiview,
    );
    let (_uniform_buffer, bind_group) = procedural_skybox_pipelines().bind_group(device, &uniforms);
    rpass.set_pipeline(pipeline.as_ref());
    rpass.set_bind_group(0, &bind_group, &[]);
    rpass.draw(0..3, 0..1);
}

fn procedural_skybox_pipelines() -> &'static ProceduralSkyPipelineCache {
    static CACHE: OnceLock<ProceduralSkyPipelineCache> = OnceLock::new();
    CACHE.get_or_init(ProceduralSkyPipelineCache::default)
}

impl ProceduralSkyUniforms {
    fn from_frame(
        frame: &FrameRenderParams<'_>,
        render_space_id: crate::scene::RenderSpaceId,
        material_id: i32,
        use_multiview: bool,
    ) -> Option<Self> {
        let matrices = view_matrices_for_skybox(frame, render_space_id, use_multiview)?;
        let property_ids =
            ProceduralSkyPropertyIds::from_registry(frame.shared.materials.property_id_registry());

        let sun_disk = material_float(frame.shared.materials.material_property_store(), material_id, property_ids.sun_disk)
            .unwrap_or(2.0)
            .round()
            .clamp(0.0, 2.0);
        let sun_size = material_float(frame.shared.materials.material_property_store(), material_id, property_ids.sun_size)
            .unwrap_or(0.04)
            .clamp(0.0, 1.0);
        let atmosphere_thickness = material_float(
            frame.shared.materials.material_property_store(),
            material_id,
            property_ids.atmosphere_thickness,
        )
        .unwrap_or(1.0)
        .clamp(0.0, 5.0);
        let exposure = material_float(
            frame.shared.materials.material_property_store(),
            material_id,
            property_ids.exposure,
        )
        .unwrap_or(1.3)
        .clamp(0.0, 8.0);

        let sun_dir = material_float4(
            frame.shared.materials.material_property_store(),
            material_id,
            property_ids.sun_direction,
        )
        .unwrap_or([0.577, 0.577, 0.577, 0.0]);
        let sun_direction = normalize_vec3_or_default(sun_dir, [0.577, 0.577, 0.577]);
        let sun_color = material_float4(
            frame.shared.materials.material_property_store(),
            material_id,
            property_ids.sun_color,
        )
        .unwrap_or([1.0, 1.0, 1.0, 1.0]);
        let sky_tint = material_float4(
            frame.shared.materials.material_property_store(),
            material_id,
            property_ids.sky_tint,
        )
        .unwrap_or([0.5, 0.5, 0.5, 1.0]);
        let ground_color = material_float4(
            frame.shared.materials.material_property_store(),
            material_id,
            property_ids.ground_color,
        )
        .unwrap_or([0.369, 0.349, 0.341, 1.0]);

        Some(Self {
            view_to_world_left: matrices.view_to_world_left.to_cols_array_2d(),
            view_to_world_right: matrices.view_to_world_right.to_cols_array_2d(),
            inv_proj_left: matrices.inv_proj_left.to_cols_array_2d(),
            inv_proj_right: matrices.inv_proj_right.to_cols_array_2d(),
            sun_direction: [
                sun_direction[0],
                sun_direction[1],
                sun_direction[2],
                0.0,
            ],
            sun_color,
            sky_tint,
            ground_color,
            params0: [sun_size, atmosphere_thickness, exposure, sun_disk],
        })
    }
}

fn view_matrices_for_skybox(
    frame: &FrameRenderParams<'_>,
    render_space_id: crate::scene::RenderSpaceId,
    use_multiview: bool,
) -> Option<ProceduralSkyViewMatrices> {
    let hc = frame.view.host_camera;
    if let (true, Some(stereo)) = (use_multiview, hc.stereo) {
        let (view_l, view_r) = stereo.view_only;
        let (vp_l, vp_r) = stereo.view_proj;
        let proj_l = vp_l * view_l.inverse();
        let proj_r = vp_r * view_r.inverse();
        return build_view_matrices(view_l, view_r, proj_l, proj_r);
    }

    if let (Some(view), Some(proj)) = (hc.cluster_view_override, hc.cluster_proj_override) {
        return build_view_matrices(view, view, proj, proj);
    }

    let space = frame.shared.scene.space(render_space_id)?;
    let view = hc
        .explicit_world_to_view
        .unwrap_or_else(|| view_matrix_for_world_mesh_render_space(frame.shared.scene, space));
    let (vw, vh) = frame.view.viewport_px;
    if vw == 0 || vh == 0 {
        return None;
    }
    let aspect = vw as f32 / vh.max(1) as f32;
    let (near_clip, far_clip) = effective_head_output_clip_planes(
        hc.near_clip,
        hc.far_clip,
        hc.output_device,
        Some(space.root_transform.scale),
    );
    let proj = reverse_z_perspective(
        aspect,
        clamp_desktop_fov_degrees(hc.desktop_fov_degrees).to_radians(),
        near_clip,
        far_clip,
    );
    build_view_matrices(view, view, proj, proj)
}

fn build_view_matrices(
    view_left: Mat4,
    view_right: Mat4,
    proj_left: Mat4,
    proj_right: Mat4,
) -> Option<ProceduralSkyViewMatrices> {
    let view_to_world_left = view_left.inverse();
    let view_to_world_right = view_right.inverse();
    let inv_proj_left = proj_left.inverse();
    let inv_proj_right = proj_right.inverse();
    [view_to_world_left, view_to_world_right, inv_proj_left, inv_proj_right]
        .into_iter()
        .all(mat4_all_finite)
        .then_some(ProceduralSkyViewMatrices {
            view_to_world_left,
            view_to_world_right,
            inv_proj_left,
            inv_proj_right,
        })
}

fn mat4_all_finite(m: Mat4) -> bool {
    m.to_cols_array().iter().all(|f| f.is_finite())
}

fn material_float(
    store: &MaterialPropertyStore,
    material_id: i32,
    property_id: Option<i32>,
) -> Option<f32> {
    let property_id = property_id?;
    match store.get_material(material_id, property_id)? {
        MaterialPropertyValue::Float(value) => Some(*value),
        _ => None,
    }
}

fn material_float4(
    store: &MaterialPropertyStore,
    material_id: i32,
    property_id: Option<i32>,
) -> Option<[f32; 4]> {
    let property_id = property_id?;
    match store.get_material(material_id, property_id)? {
        MaterialPropertyValue::Float4(value) => Some(*value),
        _ => None,
    }
}

fn normalize_vec3_or_default(value: [f32; 4], fallback: [f32; 3]) -> [f32; 3] {
    let v = glam::Vec3::new(value[0], value[1], value[2]);
    if v.length_squared() > 1.0e-8 {
        let n = v.normalize();
        [n.x, n.y, n.z]
    } else {
        fallback
    }
}

fn is_procedural_sky_shader_name(name: &str) -> bool {
    let normalized = normalize_unity_shader_lookup_key(name);
    PROCEDURAL_SKY_UNITY_KEYS
        .iter()
        .any(|candidate| *candidate == normalized)
}

#[cfg(test)]
mod tests {
    use super::is_procedural_sky_shader_name;

    #[test]
    fn procedural_sky_name_match_accepts_unity_and_resonite_forms() {
        assert!(is_procedural_sky_shader_name("ProceduralSky"));
        assert!(is_procedural_sky_shader_name("ProceduralSkyBox"));
        assert!(is_procedural_sky_shader_name("proceduralskybox"));
        assert!(!is_procedural_sky_shader_name("Custom/NotSky"));
    }
}
