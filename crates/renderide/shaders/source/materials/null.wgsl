//! Unity `Shader "Null"`: procedural object-space checkerboard with a smooth transition kernel,
//! used as a placeholder material when nothing else is bound.

// unity-shader-name: Null

#import renderide::globals as rg
#import renderide::per_draw as pd

struct NullMaterial {
    _pad: vec4<f32>,
}

@group(1) @binding(0) var<uniform> mat: NullMaterial;

const TRANSITION: f32 = 50.0;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) checker: vec3<f32>,
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) _n: vec4<f32>,
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
    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.checker = pos.xyz * 5.0;
    return out;
}

/// Transition kernel from the Unity source: maps `p in [0, 1]` to a smoothed swap factor in
/// `[0, 1]`, which the caller uses to lerp between two channel values. Returns the swapped pair.
fn transition(p_in: f32, a: f32, b: f32) -> vec2<f32> {
    var p = p_in * TRANSITION;
    if (p < TRANSITION * 0.25) {
        p = clamp(p + 0.5, 0.0, 1.0);
    } else if (p < TRANSITION * 0.75) {
        p = 1.0 - clamp(p - TRANSITION * 0.5 - 0.5, 0.0, 1.0);
    } else {
        p = 1.0 - clamp(TRANSITION - p + 0.5, 0.0, 1.0);
    }
    return vec2<f32>(mix(a, b, p), mix(b, a, p));
}

@fragment
fn fs_main(
    @location(0) checker_in: vec3<f32>,
) -> @location(0) vec4<f32> {
    let checker = fract(checker_in);
    var ab = vec2<f32>(0.0, 0.05);
    ab = transition(checker.x, ab.x, ab.y);
    ab = transition(checker.y, ab.x, ab.y);
    ab = transition(checker.z, ab.x, ab.y);
    let intensity = ab.x;
    let touch = mat._pad.x * 0.0;
    return rg::retain_globals_additive(vec4<f32>(vec3<f32>(intensity + touch), 1.0));
}
