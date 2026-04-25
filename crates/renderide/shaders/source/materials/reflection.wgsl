//! Unity shader `Shader "Reflection"`: screen-projected reflection-texture sampling.
//!
//! `_ReflectionTex` is rendered/uploaded by the host (e.g. a separate reflection-camera RT) and
//! sampled here via projective screen coordinates. Optional tangent-space normal-map perturbation
//! distorts the projected UV before the perspective divide (matching Unity's pre-divide order).
//!
//! ## Stereo packing
//!
//! Unity's convention packs both eyes side-by-side in a single 2D `_ReflectionTex` (left half
//! for the left eye, right half for the right eye). Under `MULTIVIEW` we honor that layout by
//! halving `uv.x` and offsetting by `view_index * 0.5`. In mono we sample the full texture.

// unity-shader-name: Reflection

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd

struct ReflectionMaterial {
    _Color: vec4<f32>,
    _NormalMap_ST: vec4<f32>,
    _Distort: f32,
    _Cutoff: f32,
    _NORMALMAP: f32,
    _COLOR: f32,
    _ALPHATEST: f32,
    _MUL_ALPHA_INTENSITY: f32,
    _MUL_RGB_BY_ALPHA: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(1) @binding(0) var<uniform> mat: ReflectionMaterial;
@group(1) @binding(1) var _ReflectionTex: texture_2d<f32>;
@group(1) @binding(2) var _ReflectionTex_sampler: sampler;
@group(1) @binding(3) var _NormalMap: texture_2d<f32>;
@group(1) @binding(4) var _NormalMap_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) screen_pos: vec4<f32>,
    @location(1) uv0: vec2<f32>,
    @location(2) @interpolate(flat) view_layer: u32,
}

// Equivalent of Unity's `ComputeNonStereoScreenPos(clip)`: projective screen coords whose `.xy/w`
// lands in `[0, 1]`. Y is flipped to match WebGPU's V-down convention so the sampled
// `_ReflectionTex` lines up with the framebuffer the host rendered into.
fn compute_screen_pos(clip_pos: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        (clip_pos.x + clip_pos.w) * 0.5,
        (clip_pos.w - clip_pos.y) * 0.5,
        clip_pos.z,
        clip_pos.w,
    );
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = d.model * vec4<f32>(pos.xyz, 1.0);
#ifdef MULTIVIEW
    var vp: mat4x4<f32>;
    if (view_idx == 0u) {
        vp = d.view_proj_left;
    } else {
        vp = d.view_proj_right;
    }
#else
    let vp = d.view_proj_left;
#endif
    let clip = vp * world_p;

    var out: VertexOutput;
    out.clip_pos = clip;
    out.screen_pos = compute_screen_pos(clip);
    out.uv0 = uv0;
#ifdef MULTIVIEW
    out.view_layer = view_idx;
#else
    out.view_layer = 0u;
#endif
    return out;
}

@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) screen_pos: vec4<f32>,
    @location(1) uv0: vec2<f32>,
    @location(2) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    var proj = screen_pos;

    if (uvu::kw_enabled(mat._NORMALMAP)) {
        let raw = textureSample(_NormalMap, _NormalMap_sampler, uvu::apply_st(uv0, mat._NormalMap_ST));
        let bump = nd::decode_ts_normal_with_placeholder_sample(raw, 1.0);
        // Unity perturbs the projective UV *before* the perspective divide.
        proj = vec4<f32>(proj.x + bump.x * mat._Distort, proj.y + bump.y * mat._Distort, proj.z, proj.w);
    }

    var uv = proj.xy / max(abs(proj.w), 1e-6) * sign(proj.w + select(1.0, 0.0, proj.w == 0.0));
    // (Behavior matches Unity's `uv.xy /= uv.w` — preserved sign so back-projected fragments
    // sample the same orientation as front-facing ones; the abs/select avoids divide-by-zero.)

#ifdef MULTIVIEW
    // Side-by-side stereo packing: left eye samples [0, 0.5), right eye samples [0.5, 1).
    uv.x = uv.x * 0.5 + f32(view_layer) * 0.5;
#endif

    var col = textureSample(_ReflectionTex, _ReflectionTex_sampler, uv);

    if (uvu::kw_enabled(mat._COLOR)) {
        col = col * mat._Color;
    }

    if (uvu::kw_enabled(mat._ALPHATEST)) {
        if (col.a - mat._Cutoff < 0.0) {
            discard;
        }
    }

    if (uvu::kw_enabled(mat._MUL_RGB_BY_ALPHA)) {
        col = vec4<f32>(col.rgb * col.a, col.a);
    }

    if (uvu::kw_enabled(mat._MUL_ALPHA_INTENSITY)) {
        let intensity = (col.r + col.g + col.b) * (1.0 / 3.0);
        col.a = col.a * intensity;
    }

    return rg::retain_globals_additive(col);
}
