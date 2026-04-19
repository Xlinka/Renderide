//! Filters/Blur_PerObject: per-object grab-pass blur with optional spread texture and refraction.
//!
//! `_GrabPass` is a marker field used by Rust reflection to route these draws through the
//! per-object scene-color snapshot path before `fs_main` samples `renderide::globals::scene_color`.

// unity-shader-name: Filters/Blur_PerObject
//#pass main: depth=greater_equal, zwrite=true, cull=back, blend=none, material=base

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::normal_decode as nd
#import renderide::uv_utils as uvu

struct BlurPerObjectMaterial {
    _Spread: vec4<f32>,
    _SpreadTex_ST: vec4<f32>,
    _NormalMap_ST: vec4<f32>,
    _Rect: vec4<f32>,
    _Iterations: f32,
    _RefractionStrength: f32,
    _DepthDivisor: f32,
    _GrabPass: f32,
    _SrcBlend: f32,
    _DstBlend: f32,
    _ZWrite: f32,
    _Cull: f32,
    _ZTest: f32,
    _OffsetFactor: f32,
    _OffsetUnits: f32,
    _StencilComp: f32,
    _Stencil: f32,
    _StencilOp: f32,
    _StencilWriteMask: f32,
    _StencilReadMask: f32,
    _ColorMask: f32,
    RECTCLIP: f32,
    REFRACT: f32,
    REFRACT_NORMALMAP: f32,
    POISSON_DISC: f32,
}

@group(1) @binding(0) var<uniform> mat: BlurPerObjectMaterial;
@group(1) @binding(1) var _SpreadTex: texture_2d<f32>;
@group(1) @binding(2) var _SpreadTex_sampler: sampler;
@group(1) @binding(3) var _NormalMap: texture_2d<f32>;
@group(1) @binding(4) var _NormalMap_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) local_xy: vec2<f32>,
    @location(2) world_normal: vec3<f32>,
    @location(3) world_tangent: vec4<f32>,
    @location(4) eye_depth: f32,
    @location(5) @interpolate(flat) view_layer: u32,
}

fn safe_normalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let l2 = dot(v, v);
    if (l2 <= 1e-10) {
        return fallback;
    }
    return v * inverseSqrt(l2);
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(4) tangent: vec4<f32>,
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
    let view_idx = 0u;
#endif

    let world_n = safe_normalize(d.normal_matrix * normal.xyz, vec3<f32>(0.0, 1.0, 0.0));
    let tangent_world = (d.model * vec4<f32>(tangent.xyz, 0.0)).xyz;

    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.uv = uv;
    out.local_xy = pos.xy;
    out.world_normal = world_n;
    out.world_tangent = vec4<f32>(safe_normalize(tangent_world, vec3<f32>(1.0, 0.0, 0.0)), tangent.w);
    out.eye_depth = max(distance(world_p.xyz, rg::frame.camera_world_pos.xyz), 1e-4);
    out.view_layer = view_idx;
    return out;
}

fn layer_sample(uv: vec2<f32>, layer: u32) -> vec4<f32> {
#ifdef MULTIVIEW
    return textureSample(rg::scene_color_array, rg::scene_color_sampler, uv, i32(layer));
#else
    return textureSample(rg::scene_color, rg::scene_color_sampler, uv);
#endif
}

fn poisson_offset(idx: u32) -> vec2<f32> {
    switch idx {
        case 0u: { return vec2<f32>(-0.326, -0.406); }
        case 1u: { return vec2<f32>(-0.840, -0.074); }
        case 2u: { return vec2<f32>(-0.696, 0.457); }
        case 3u: { return vec2<f32>(-0.203, 0.621); }
        case 4u: { return vec2<f32>(0.963, -0.195); }
        case 5u: { return vec2<f32>(0.473, -0.480); }
        case 6u: { return vec2<f32>(0.519, 0.767); }
        default: { return vec2<f32>(0.185, -0.893); }
    }
}

fn ring_offset(idx: u32) -> vec2<f32> {
    switch idx & 7u {
        case 0u: { return vec2<f32>(1.0, 0.0); }
        case 1u: { return vec2<f32>(0.7071, 0.7071); }
        case 2u: { return vec2<f32>(0.0, 1.0); }
        case 3u: { return vec2<f32>(-0.7071, 0.7071); }
        case 4u: { return vec2<f32>(-1.0, 0.0); }
        case 5u: { return vec2<f32>(-0.7071, -0.7071); }
        case 6u: { return vec2<f32>(0.0, -1.0); }
        default: { return vec2<f32>(0.7071, -0.7071); }
    }
}

fn screen_uv_from_position(pos: vec4<f32>) -> vec2<f32> {
    let size = vec2<f32>(f32(max(rg::frame.viewport_width, 1u)), f32(max(rg::frame.viewport_height, 1u)));
    return clamp(pos.xy / size, vec2<f32>(0.0), vec2<f32>(1.0));
}

fn refracted_uv(base_uv: vec2<f32>, in: VertexOutput) -> vec2<f32> {
    if (mat.REFRACT <= 0.5 && mat.REFRACT_NORMALMAP <= 0.5) {
        return base_uv;
    }

    var n = safe_normalize(in.world_normal, vec3<f32>(0.0, 0.0, 1.0));
    if (mat.REFRACT_NORMALMAP > 0.5) {
        let t = safe_normalize(in.world_tangent.xyz - n * dot(in.world_tangent.xyz, n), vec3<f32>(1.0, 0.0, 0.0));
        let sign = select(1.0, -1.0, in.world_tangent.w < 0.0);
        let b = safe_normalize(cross(n, t) * sign, vec3<f32>(0.0, 0.0, 1.0));
        let ts = nd::decode_ts_normal_with_placeholder_sample(
            textureSample(_NormalMap, _NormalMap_sampler, uvu::apply_st(in.uv, mat._NormalMap_ST)),
            1.0,
        );
        n = safe_normalize(mat3x3<f32>(t, b, n) * ts, n);
    }

    return clamp(base_uv - n.xy * mat._RefractionStrength, vec2<f32>(0.0), vec2<f32>(1.0));
}

fn blur_scene(grab_uv: vec2<f32>, mesh_uv: vec2<f32>, eye_depth: f32, layer: u32) -> vec4<f32> {
    let spread_uv = max(mat._Spread.xy, vec2<f32>(0.0));
    let spread_tex = textureSample(_SpreadTex, _SpreadTex_sampler, uvu::apply_st(mesh_uv, mat._SpreadTex_ST)).rg;
    let depth_scale = select(1.0, clamp(mat._DepthDivisor / max(eye_depth, 1e-4), 0.0, 1.0), mat._DepthDivisor > 1e-5);
    let step_uv = spread_uv * max(spread_tex, vec2<f32>(0.0)) * depth_scale;
    let iterations = u32(clamp(round(mat._Iterations), 1.0, 12.0));
    let poisson = mat.POISSON_DISC > 0.5;

    var sum = layer_sample(grab_uv, layer);
    var weight_sum = 1.0;
    var r = 1u;
    loop {
        if (r > iterations) {
            break;
        }
        let radius = f32(r) / f32(iterations);
        var s = 0u;
        loop {
            if (s >= 8u) {
                break;
            }
            let dir = select(ring_offset(s), poisson_offset(s), poisson);
            let offset = dir * step_uv * radius;
            let weight = 1.0 - radius * 0.45;
            sum = sum + layer_sample(clamp(grab_uv + offset, vec2<f32>(0.0), vec2<f32>(1.0)), layer) * weight;
            sum = sum + layer_sample(clamp(grab_uv - offset, vec2<f32>(0.0), vec2<f32>(1.0)), layer) * weight;
            weight_sum = weight_sum + weight * 2.0;
            s = s + 1u;
        }
        r = r + 1u;
    }
    return vec4<f32>((sum / weight_sum).rgb, 1.0 + mat._GrabPass * 0.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (mat.RECTCLIP > 0.5) {
        if (in.local_xy.x < mat._Rect.x || in.local_xy.y < mat._Rect.y ||
            in.local_xy.x > mat._Rect.z || in.local_xy.y > mat._Rect.w) {
            discard;
        }
    }

    let grab_uv = refracted_uv(screen_uv_from_position(in.clip_pos), in);
    let color = blur_scene(grab_uv, in.uv, in.eye_depth, in.view_layer);
    return rg::retain_globals_additive(color);
}
