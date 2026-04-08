//! Debug mesh material: world-space normals as RGB (`shaders/target/debug_world_normals_*.wgsl`).

use crate::materials::raster_pipeline::create_reflective_raster_mesh_forward_pipeline;
use crate::materials::{reflect_raster_material_wgsl, validate_per_draw_group2};
use crate::materials::{MaterialFamilyId, MaterialPipelineDesc, MaterialPipelineFamily};
use crate::pipelines::ShaderPermutation;

/// Builtin family id for [`DebugWorldNormalsFamily`].
pub const DEBUG_WORLD_NORMALS_FAMILY_ID: MaterialFamilyId = MaterialFamilyId(2);

/// [`ShaderPermutation`] for multiview WGSL (`debug_world_normals_multiview` target stem).
pub const SHADER_PERM_MULTIVIEW_STEREO: ShaderPermutation = ShaderPermutation(1);

/// World-normal debug visualization for decomposed position/normal vertex streams.
pub struct DebugWorldNormalsFamily;

impl DebugWorldNormalsFamily {
    /// `@group(2)` dynamic uniform layout for [`crate::backend::DebugDrawResources`].
    ///
    /// Matches naga reflection of the embedded `debug_world_normals_default` target (same `@group(2)`
    /// as the multiview variant).
    pub fn per_draw_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let wgsl = crate::embedded_shaders::embedded_target_wgsl("debug_world_normals_default")
            .expect("embedded debug_world_normals_default");
        let r = reflect_raster_material_wgsl(wgsl).expect("reflect per_draw layout");
        validate_per_draw_group2(&r.per_draw_entries).expect("per_draw group2");
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("debug_world_normals_per_draw"),
            entries: &r.per_draw_entries,
        })
    }

    fn target_stem(permutation: ShaderPermutation) -> &'static str {
        if permutation.0 == SHADER_PERM_MULTIVIEW_STEREO.0 {
            "debug_world_normals_multiview"
        } else {
            "debug_world_normals_default"
        }
    }
}

impl MaterialPipelineFamily for DebugWorldNormalsFamily {
    fn family_id(&self) -> MaterialFamilyId {
        DEBUG_WORLD_NORMALS_FAMILY_ID
    }

    fn build_wgsl(&self, permutation: ShaderPermutation) -> String {
        let stem = Self::target_stem(permutation);
        crate::embedded_shaders::embedded_target_wgsl(stem)
            .unwrap_or_else(|| {
                panic!("composed shader missing for stem {stem} (run build with shaders/source)")
            })
            .to_string()
    }

    fn create_render_pipeline(
        &self,
        device: &wgpu::Device,
        module: &wgpu::ShaderModule,
        desc: &MaterialPipelineDesc,
        wgsl_source: &str,
    ) -> wgpu::RenderPipeline {
        create_reflective_raster_mesh_forward_pipeline(
            device,
            module,
            desc,
            wgsl_source,
            "debug_world_normals_material",
            false,
        )
    }
}
