//! Per-pass pipeline descriptor for multi-pass material shaders.
//!
//! A material stem may declare multiple passes via `//#pass` directives parsed in `build.rs` and
//! embedded alongside the composed WGSL (see [`crate::embedded_shaders::embedded_target_passes`]).
//! Each pass becomes one `wgpu::RenderPipeline`; the forward encode loop dispatches all pipelines
//! for every draw that binds the material, in declared order.
//!
//! Single-pass materials (the majority) do not declare any directives and fall through to
//! [`default_pass`], which preserves the pre-multi-pass behavior exactly.

use crate::assets::material::{
    MaterialDictionary, MaterialPropertyLookupIds, MaterialPropertyValue, PropertyIdRegistry,
};

/// Resonite/Froox material blend mode, or the shader stem's default when no material field is present.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MaterialBlendMode {
    /// No material-level override; use the stem's normal static behavior.
    #[default]
    StemDefault,
    /// `BlendMode.Opaque` (`0`).
    Opaque,
    /// `BlendMode.Cutout` (`1`).
    Cutout,
    /// `BlendMode.Alpha` (`2`).
    Alpha,
    /// `BlendMode.Transparent` (`3`).
    Transparent,
    /// `BlendMode.Additive` (`4`).
    Additive,
    /// `BlendMode.Multiply` (`5`).
    Multiply,
}

impl MaterialBlendMode {
    /// Converts Resonite's `BlendMode` enum value into a pipeline mode.
    pub fn from_resonite_value(v: f32) -> Self {
        match v.round() as i32 {
            0 => Self::Opaque,
            1 => Self::Cutout,
            2 => Self::Alpha,
            3 => Self::Transparent,
            4 => Self::Additive,
            5 => Self::Multiply,
            _ => Self::StemDefault,
        }
    }

    /// Returns true when the mode must be sorted/drawn as transparent.
    pub fn is_transparent(self) -> bool {
        matches!(
            self,
            Self::Alpha | Self::Transparent | Self::Additive | Self::Multiply
        )
    }
}

/// Property ids used for material-driven pipeline state.
#[derive(Clone, Copy, Debug)]
pub struct MaterialPipelinePropertyIds {
    blend_mode: [i32; 2],
}

impl MaterialPipelinePropertyIds {
    /// Interns property names used by Resonite material components and Unity-style shaders.
    pub fn new(registry: &PropertyIdRegistry) -> Self {
        Self {
            blend_mode: [registry.intern("_BlendMode"), registry.intern("BlendMode")],
        }
    }
}

/// Resolves a material/property-block `BlendMode` override.
pub fn material_blend_mode_for_lookup(
    dict: &MaterialDictionary<'_>,
    lookup: MaterialPropertyLookupIds,
    ids: &MaterialPipelinePropertyIds,
) -> MaterialBlendMode {
    for pid in ids.blend_mode {
        if let Some(MaterialPropertyValue::Float(f)) = dict.get_merged(lookup, pid) {
            return MaterialBlendMode::from_resonite_value(*f);
        }
    }
    MaterialBlendMode::StemDefault
}

/// Pipeline state for one pass of a material shader. All fields are `const`-constructible so the
/// build script can emit tables directly into generated Rust.
#[derive(Debug, Clone, Copy)]
pub struct MaterialPassDesc {
    /// Debug label for logs / pipeline names.
    pub name: &'static str,
    /// Vertex shader entry point.
    pub vertex_entry: &'static str,
    /// Fragment shader entry point.
    pub fragment_entry: &'static str,
    /// Depth comparison under reverse-Z. Unity `LEqual` maps to `GreaterEqual`; Unity `Greater` maps to `Less`.
    pub depth_compare: wgpu::CompareFunction,
    /// Whether this pass writes to the depth buffer.
    pub depth_write: bool,
    /// Backface culling mode (`None` = disabled).
    pub cull_mode: Option<wgpu::Face>,
    /// Color + alpha blend state, or `None` for no blending.
    pub blend: Option<wgpu::BlendState>,
    /// Color attachment write mask.
    pub write_mask: wgpu::ColorWrites,
    /// Slope-scaled depth bias.
    pub depth_bias_slope_scale: f32,
    /// Constant depth bias.
    pub depth_bias_constant: i32,
}

/// Default single-pass descriptor matching the pre-multi-pass pipeline builder.
///
/// `use_alpha_blending` picks between opaque (`ColorWrites::COLOR`, `blend: None`) and transparent
/// (`ColorWrites::ALL`, `ALPHA_BLENDING`). `depth_write` mirrors the old `depth_write_enabled` arg.
pub const fn default_pass(use_alpha_blending: bool, depth_write: bool) -> MaterialPassDesc {
    let (blend, write_mask) = if use_alpha_blending {
        (
            Some(wgpu::BlendState::ALPHA_BLENDING),
            wgpu::ColorWrites::ALL,
        )
    } else {
        (None, wgpu::ColorWrites::COLOR)
    };
    MaterialPassDesc {
        name: "main",
        vertex_entry: "vs_main",
        fragment_entry: "fs_main",
        depth_compare: crate::render_graph::MAIN_FORWARD_DEPTH_COMPARE,
        depth_write,
        cull_mode: None,
        blend,
        write_mask,
        depth_bias_slope_scale: 0.0,
        depth_bias_constant: 0,
    }
}

/// Default single-pass descriptor after applying a material `BlendMode` override.
pub const fn default_pass_for_blend_mode(
    stem_uses_alpha_blending: bool,
    blend_mode: MaterialBlendMode,
) -> MaterialPassDesc {
    match blend_mode {
        MaterialBlendMode::StemDefault => {
            default_pass(stem_uses_alpha_blending, !stem_uses_alpha_blending)
        }
        MaterialBlendMode::Opaque | MaterialBlendMode::Cutout => default_pass(false, true),
        MaterialBlendMode::Alpha | MaterialBlendMode::Transparent => default_pass(true, false),
        MaterialBlendMode::Additive => MaterialPassDesc {
            name: "additive",
            vertex_entry: "vs_main",
            fragment_entry: "fs_main",
            depth_compare: crate::render_graph::MAIN_FORWARD_DEPTH_COMPARE,
            depth_write: false,
            cull_mode: None,
            blend: Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Max,
                },
            }),
            write_mask: wgpu::ColorWrites::ALL,
            depth_bias_slope_scale: 0.0,
            depth_bias_constant: 0,
        },
        MaterialBlendMode::Multiply => MaterialPassDesc {
            name: "multiply",
            vertex_entry: "vs_main",
            fragment_entry: "fs_main",
            depth_compare: crate::render_graph::MAIN_FORWARD_DEPTH_COMPARE,
            depth_write: false,
            cull_mode: None,
            blend: Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Dst,
                    dst_factor: wgpu::BlendFactor::Zero,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
            write_mask: wgpu::ColorWrites::ALL,
            depth_bias_slope_scale: 0.0,
            depth_bias_constant: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::material::{MaterialPropertyStore, PropertyIdRegistry};

    #[test]
    fn resolves_resonite_blend_mode_property() {
        let reg = PropertyIdRegistry::new();
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut store = MaterialPropertyStore::new();
        let pid = reg.intern("BlendMode");
        store.set_material(42, pid, MaterialPropertyValue::Float(4.0));
        let dict = MaterialDictionary::new(&store);
        let lookup = MaterialPropertyLookupIds {
            material_asset_id: 42,
            mesh_property_block_slot0: None,
        };
        assert_eq!(
            material_blend_mode_for_lookup(&dict, lookup, &ids),
            MaterialBlendMode::Additive
        );
    }
}
