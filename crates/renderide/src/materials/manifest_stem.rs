//! Convention-based raster materials: one [`MaterialFamilyId`] and many composed WGSL stems.
//!
//! [`ManifestStemMaterialFamily`] is constructed per resolved stem (see [`super::MaterialRegistry`]).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::embedded_shaders;
use crate::materials::raster_pipeline::create_reflective_raster_mesh_forward_pipeline;
use crate::materials::{MaterialFamilyId, MaterialPipelineDesc, MaterialPipelineFamily};
use crate::pipelines::raster::SHADER_PERM_MULTIVIEW_STEREO;
use crate::pipelines::ShaderPermutation;

/// Stable id for shaders whose normalized Unity name has an embedded `{key}_default` WGSL target.
pub const MANIFEST_RASTER_FAMILY_ID: MaterialFamilyId = MaterialFamilyId(3);

fn manifest_uv0_stream_cache() -> &'static Mutex<HashMap<String, bool>> {
    static CACHE: OnceLock<Mutex<HashMap<String, bool>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// `true` when composed manifest WGSL's `vs_main` uses `@location(2)` or higher (UV0 vertex stream).
///
/// Uses the same embedded source and reflection as [`ManifestStemMaterialFamily::create_render_pipeline`]
/// for the given [`ShaderPermutation`], independent of [`crate::backend::ManifestMaterialBindResources`].
///
/// Results are memoized per `(base_stem, permutation)` so draw collection and other hot paths do not
/// re-run naga reflection once per mesh draw.
pub fn manifest_stem_needs_uv0_stream(base_stem: &str, permutation: ShaderPermutation) -> bool {
    let key = format!("{base_stem}:{}", permutation.0);
    let mut guard = manifest_uv0_stream_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(v) = guard.get(&key) {
        return *v;
    }
    let composed = manifest_composed_stem_for_permutation(base_stem, permutation);
    let v = embedded_shaders::embedded_target_wgsl(&composed)
        .map(manifest_wgsl_needs_uv0_stream)
        .unwrap_or(false);
    guard.insert(key, v);
    v
}

/// `true` when `vs_main` reflection reports a highest vertex `@location` index ≥ 2 (UV at `location(2)`).
pub fn manifest_wgsl_needs_uv0_stream(wgsl_source: &str) -> bool {
    crate::materials::wgsl_reflect::reflect_vertex_shader_needs_uv0_stream(wgsl_source)
}

/// Composed target stem for a manifest base stem (e.g. `unlit_default` → `unlit_multiview`).
pub fn manifest_composed_stem_for_permutation(
    base_stem: &str,
    permutation: ShaderPermutation,
) -> String {
    if permutation.0 == SHADER_PERM_MULTIVIEW_STEREO.0 {
        if base_stem.ends_with("_default") {
            return format!("{}_multiview", base_stem.trim_end_matches("_default"));
        }
        return base_stem.to_string();
    }
    if base_stem.ends_with("_multiview") {
        return format!("{}_default", base_stem.trim_end_matches("_multiview"));
    }
    base_stem.to_string()
}

/// Raster family parameterized by a manifest stem (`shaders/target/<composed_stem>.wgsl`).
#[derive(Debug)]
pub struct ManifestStemMaterialFamily {
    /// Stem from [`super::MaterialRouter::stem_for_shader_asset`] (e.g. `unlit_default`).
    pub stem: Arc<str>,
}

impl ManifestStemMaterialFamily {
    /// Builds a family for the given manifest stem.
    pub fn new(stem: Arc<str>) -> Self {
        Self { stem }
    }

    fn composed_stem(&self, permutation: ShaderPermutation) -> String {
        manifest_composed_stem_for_permutation(self.stem.as_ref(), permutation)
    }
}

impl MaterialPipelineFamily for ManifestStemMaterialFamily {
    fn family_id(&self) -> MaterialFamilyId {
        MANIFEST_RASTER_FAMILY_ID
    }

    fn manifest_stem(&self) -> Option<Arc<str>> {
        Some(self.stem.clone())
    }

    fn build_wgsl(&self, permutation: ShaderPermutation) -> String {
        let stem = self.composed_stem(permutation);
        embedded_shaders::embedded_target_wgsl(&stem)
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
            "manifest_stem_material",
            true,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipelines::ShaderPermutation;
    use crate::pipelines::SHADER_PERM_MULTIVIEW_STEREO;

    #[test]
    fn debug_world_normals_no_uv0_stream() {
        assert!(!manifest_stem_needs_uv0_stream(
            "debug_world_normals_default",
            ShaderPermutation(0)
        ));
        assert!(!manifest_stem_needs_uv0_stream(
            "debug_world_normals_default",
            SHADER_PERM_MULTIVIEW_STEREO
        ));
    }
}
