//! Resolves [`ShaderUpload`](crate::shared::ShaderUpload) to a [`MaterialFamilyId`] for [`MaterialRegistry`](crate::materials::MaterialRegistry).
//!
//! Extraction of Unity logical names lives in [`super::logical_name`] and [`super::unity_asset`].
//! [`resolve_shader_upload`] uses
//! [`super::logical_name::resolve_shader_routing_name_from_upload`] so filesystem paths prefer raw
//! AssetBundle / container stems before ShaderLab first-token canonicalization.
//!
//! Names with an embedded `{logical}_default` WGSL target (see [`crate::materials::stem_manifest`]) resolve to
//! [`MANIFEST_RASTER_FAMILY_ID`](crate::materials::MANIFEST_RASTER_FAMILY_ID); unknown or non-embedded shaders use
//! [`DEBUG_WORLD_NORMALS_FAMILY_ID`](crate::materials::DEBUG_WORLD_NORMALS_FAMILY_ID) as the **only** mesh fallback
//! (there is no separate solid-color pipeline family).

use crate::materials::DEBUG_WORLD_NORMALS_FAMILY_ID;
use crate::materials::{manifest_stem_for_unity_name, MaterialFamilyId, MANIFEST_RASTER_FAMILY_ID};
use crate::shared::ShaderUpload;

use super::logical_name;

/// Resolved upload: optional Unity-style logical name plus the material family for pipeline selection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedShaderUpload {
    /// `Shader "…"` string, container stem, or label when resolution succeeded.
    pub unity_shader_name: Option<String>,
    /// Family passed to [`crate::materials::MaterialRegistry::map_shader_route`].
    pub family: MaterialFamilyId,
}

/// Full resolution pipeline for a host [`ShaderUpload`].
pub fn resolve_shader_upload(data: &ShaderUpload) -> ResolvedShaderUpload {
    let unity_shader_name = logical_name::resolve_shader_routing_name_from_upload(data, None);
    let family = match unity_shader_name.as_deref() {
        Some(name) if manifest_stem_for_unity_name(name).is_some() => MANIFEST_RASTER_FAMILY_ID,
        _ => DEBUG_WORLD_NORMALS_FAMILY_ID,
    };
    ResolvedShaderUpload {
        unity_shader_name,
        family,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::DEBUG_WORLD_NORMALS_FAMILY_ID;

    #[test]
    fn shader_lab_unlit_resolves_manifest_family() {
        let u = ShaderUpload {
            asset_id: 1,
            file: Some("Shader \"Unlit\"\n{\n".to_string()),
        };
        let r = resolve_shader_upload(&u);
        assert_eq!(r.family, MANIFEST_RASTER_FAMILY_ID);
    }

    #[test]
    fn unknown_shader_uses_debug_family() {
        let u = ShaderUpload {
            asset_id: 2,
            file: Some("Shader \"Custom/NoSuchEmbeddedShader\"\n{\n".to_string()),
        };
        let r = resolve_shader_upload(&u);
        assert_eq!(r.family, DEBUG_WORLD_NORMALS_FAMILY_ID);
    }
}
