//! Material-driven raster state resolved from Unity-style properties (`_Stencil`, `_ZWrite`, `_Cull`, …).
//!
//! Used by the mesh-forward draw prep path and reflective raster pipeline builders to key
//! [`wgpu::RenderPipeline`] instances consistently with host material overrides.

use crate::assets::material::{MaterialDictionary, MaterialPropertyLookupIds};

use super::material_pass_tables::{
    unity_color_writes, unity_compare_function, unity_depth_compare_function,
    unity_stencil_operation,
};
use super::material_passes::{first_float_by_pids, MaterialPipelinePropertyIds};

/// Unity `Cull` / `CullMode` material override for raster pipeline keys and [`MaterialRenderState::resolved_cull_mode`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MaterialCullOverride {
    /// No `_Cull` / `_Culling` property (or unknown enum value): use the pass default.
    #[default]
    Unspecified,
    /// `Cull Off` — disable backface culling.
    Off,
    /// `Cull Front`.
    Front,
    /// `Cull Back`.
    Back,
}

/// Runtime Unity stencil/color/depth/cull state resolved from material properties.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaterialRenderState {
    /// Stencil state for this draw. Disabled when no stencil-related material property is present.
    pub stencil: MaterialStencilState,
    /// Unity `ColorMask` override. `None` preserves the shader pass default.
    pub color_mask: Option<u8>,
    /// Unity `ZWrite` override. `None` preserves the shader pass default.
    pub depth_write: Option<bool>,
    /// Unity `ZTest` / `CompareFunction` override. `None` preserves the shader pass default.
    pub depth_compare: Option<u8>,
    /// Unity `Offset factor, units` override. `None` preserves the shader pass default.
    pub depth_offset: Option<MaterialDepthOffsetState>,
    /// Unity `Cull` / `_Culling` override for wgpu [`PrimitiveState::cull_mode`](wgpu::PrimitiveState::cull_mode).
    pub cull_override: MaterialCullOverride,
}

/// Unity `Offset factor, units` state stored in an ordered/hashable form for pipeline keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaterialDepthOffsetState {
    factor_bits: u32,
    units: i32,
}

impl MaterialDepthOffsetState {
    /// Creates non-zero Unity `Offset factor, units` state for a material pipeline key.
    pub fn new(factor: f32, units: i32) -> Option<Self> {
        let factor = if factor.is_finite() { factor } else { 0.0 };
        let factor = if factor == 0.0 { 0.0 } else { factor };
        if factor == 0.0 && units == 0 {
            return None;
        }
        Some(Self {
            factor_bits: factor.to_bits(),
            units,
        })
    }

    /// Unity slope-scaled offset factor as raw bits for ordered/hashable diagnostics.
    pub fn factor_bits(self) -> u32 {
        self.factor_bits
    }

    /// Unity slope-scaled offset factor.
    pub fn factor(self) -> f32 {
        f32::from_bits(self.factor_bits)
    }

    /// Unity constant offset units.
    pub fn units(self) -> i32 {
        self.units
    }
}

impl MaterialRenderState {
    /// Stencil reference passed via dynamic render pass state.
    pub fn stencil_reference(self) -> u32 {
        self.stencil.reference
    }

    /// Applies the optional Unity color-mask override to a pass write mask.
    pub fn color_writes(self, fallback: wgpu::ColorWrites) -> wgpu::ColorWrites {
        self.color_mask.map(unity_color_writes).unwrap_or(fallback)
    }

    /// Applies the optional Unity depth-write override to a pass default.
    pub fn depth_write(self, fallback: bool) -> bool {
        self.depth_write.unwrap_or(fallback)
    }

    /// Applies the optional Unity depth-compare override to a pass default.
    pub fn depth_compare(self, fallback: wgpu::CompareFunction) -> wgpu::CompareFunction {
        self.depth_compare
            .and_then(unity_depth_compare_function)
            .unwrap_or(fallback)
    }

    /// Applies [`Self::cull_override`] to a pass default (`None` = culling disabled).
    pub fn resolved_cull_mode(self, fallback: Option<wgpu::Face>) -> Option<wgpu::Face> {
        match self.cull_override {
            MaterialCullOverride::Unspecified => fallback,
            MaterialCullOverride::Off => None,
            MaterialCullOverride::Front => Some(wgpu::Face::Front),
            MaterialCullOverride::Back => Some(wgpu::Face::Back),
        }
    }

    /// Applies Unity `Offset` to wgpu depth bias, accounting for reverse-Z.
    pub fn depth_bias(
        self,
        fallback_constant: i32,
        fallback_slope_scale: f32,
    ) -> wgpu::DepthBiasState {
        match self.depth_offset {
            Some(offset) => wgpu::DepthBiasState {
                constant: offset.units().saturating_neg(),
                slope_scale: -offset.factor(),
                clamp: 0.0,
            },
            None => wgpu::DepthBiasState {
                constant: fallback_constant,
                slope_scale: fallback_slope_scale,
                clamp: 0.0,
            },
        }
    }

    /// Converts the resolved material state into a wgpu stencil state.
    pub fn stencil_state(self) -> wgpu::StencilState {
        if !self.stencil.enabled {
            return wgpu::StencilState::default();
        }
        let face = wgpu::StencilFaceState {
            compare: unity_compare_function(self.stencil.compare),
            fail_op: unity_stencil_operation(self.stencil.fail_op),
            depth_fail_op: unity_stencil_operation(self.stencil.depth_fail_op),
            pass_op: unity_stencil_operation(self.stencil.pass_op),
        };
        wgpu::StencilState {
            front: face,
            back: face,
            read_mask: self.stencil.read_mask,
            write_mask: self.stencil.write_mask,
        }
    }
}

/// Unity-compatible stencil material state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaterialStencilState {
    /// Whether the stencil test/write path should be enabled for this draw.
    pub enabled: bool,
    /// Dynamic stencil reference value.
    pub reference: u32,
    /// Unity `CompareFunction` enum value.
    pub compare: u8,
    /// Unity `StencilOp` enum value applied on pass.
    pub pass_op: u8,
    /// Unity `StencilOp` enum value applied when stencil comparison fails.
    pub fail_op: u8,
    /// Unity `StencilOp` enum value applied when depth comparison fails.
    pub depth_fail_op: u8,
    /// Stencil read mask.
    pub read_mask: u32,
    /// Stencil write mask.
    pub write_mask: u32,
}

impl Default for MaterialStencilState {
    fn default() -> Self {
        Self {
            enabled: false,
            reference: 0,
            compare: 8,
            pass_op: 0,
            fail_op: 0,
            depth_fail_op: 0,
            read_mask: 0xff,
            write_mask: 0xff,
        }
    }
}

fn unity_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

fn unity_mask(v: f32) -> u32 {
    v.round().clamp(0.0, 255.0) as u32
}

fn unity_offset_units(v: f32) -> i32 {
    v.round().clamp(i32::MIN as f32, i32::MAX as f32) as i32
}

/// Resolves Unity color, stencil, and depth properties for a material/property-block pair.
pub fn material_render_state_for_lookup(
    dict: &MaterialDictionary<'_>,
    lookup: MaterialPropertyLookupIds,
    ids: &MaterialPipelinePropertyIds,
) -> MaterialRenderState {
    let stencil_ref = first_float_by_pids(dict, lookup, &ids.stencil_ref);
    let stencil_comp = first_float_by_pids(dict, lookup, &ids.stencil_comp);
    let stencil_op = first_float_by_pids(dict, lookup, &ids.stencil_op);
    let stencil_fail_op = first_float_by_pids(dict, lookup, &ids.stencil_fail_op);
    let stencil_depth_fail_op = first_float_by_pids(dict, lookup, &ids.stencil_depth_fail_op);
    let stencil_read_mask = first_float_by_pids(dict, lookup, &ids.stencil_read_mask);
    let stencil_write_mask = first_float_by_pids(dict, lookup, &ids.stencil_write_mask);
    let color_mask = first_float_by_pids(dict, lookup, &ids.color_mask).map(unity_u8);
    let depth_write =
        first_float_by_pids(dict, lookup, &ids.z_write).map(|v| v.round().clamp(0.0, 1.0) >= 0.5);
    let depth_compare = first_float_by_pids(dict, lookup, &ids.z_test).map(unity_u8);
    let cull_override = match first_float_by_pids(dict, lookup, &ids.cull).map(unity_u8) {
        None => MaterialCullOverride::Unspecified,
        // UnityEngine.Rendering.CullMode: Off / Front / Back
        Some(0) => MaterialCullOverride::Off,
        Some(1) => MaterialCullOverride::Front,
        Some(2) => MaterialCullOverride::Back,
        Some(_) => MaterialCullOverride::Unspecified,
    };
    let depth_offset = {
        let factor = first_float_by_pids(dict, lookup, &ids.offset_factor);
        let units = first_float_by_pids(dict, lookup, &ids.offset_units);
        if factor.is_some() || units.is_some() {
            MaterialDepthOffsetState::new(
                factor.unwrap_or(0.0),
                units.map(unity_offset_units).unwrap_or(0),
            )
        } else {
            None
        }
    };

    let stencil_present = stencil_ref.is_some()
        || stencil_comp.is_some()
        || stencil_op.is_some()
        || stencil_fail_op.is_some()
        || stencil_depth_fail_op.is_some()
        || stencil_read_mask.is_some()
        || stencil_write_mask.is_some();
    let compare = stencil_comp.map(unity_u8).unwrap_or(8);
    let stencil = MaterialStencilState {
        enabled: stencil_present && compare != 0,
        reference: stencil_ref.map(unity_mask).unwrap_or(0),
        compare,
        pass_op: stencil_op.map(unity_u8).unwrap_or(0),
        fail_op: stencil_fail_op.map(unity_u8).unwrap_or(0),
        depth_fail_op: stencil_depth_fail_op.map(unity_u8).unwrap_or(0),
        read_mask: stencil_read_mask.map(unity_mask).unwrap_or(0xff),
        write_mask: stencil_write_mask.map(unity_mask).unwrap_or(0xff),
    };

    MaterialRenderState {
        stencil,
        color_mask,
        depth_write,
        depth_compare,
        depth_offset,
        cull_override,
    }
}
