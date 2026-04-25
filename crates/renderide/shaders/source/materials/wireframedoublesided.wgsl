//! Unity wireframe double-sided (`Shader "WireframeDoubleSided"`).
//!
//! Unity's source uses a geometry shader to emit per-triangle edge-distance varyings and splits
//! front/back rendering into two raster passes. WGSL has no geometry stage, so Renderide renders
//! this as a single alpha-blended cull-off pass using `front_facing` to select the outer vs inner
//! color set. A lazily built triangle-expanded mesh cache supplies barycentrics plus per-triangle
//! edge altitudes for the object-space thickness mode.

// unity-shader-name: WireframeDoubleSided

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::uv_utils as uvu
#import renderide::texture_sampling as ts

struct WireframeDoubleSidedMaterial {
    _LineColor: vec4<f32>,
    _FillColor: vec4<f32>,
    _InnerLineColor: vec4<f32>,
    _InnerFillColor: vec4<f32>,
    _LineFarColor: vec4<f32>,
    _FillFarColor: vec4<f32>,
    _InnerLineFarColor: vec4<f32>,
    _InnerFillFarColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _Exp: f32,
    _Thickness: f32,
    _SCREENSPACE: f32,
    _FRESNEL: f32,
    _MainTex_LodBias: f32,
}

@group(1) @binding(0) var<uniform> mat: WireframeDoubleSidedMaterial;
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
@group(1) @binding(2) var _MainTex_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) bary: vec3<f32>,
    @location(4) object_edge_dist: vec3<f32>,
}

fn kw(v: f32) -> bool {
    return v > 0.5;
}

fn max_component_scale(model: mat4x4<f32>) -> f32 {
    return max(
        length(model[0].xyz),
        max(length(model[1].xyz), length(model[2].xyz)),
    );
}

fn min3(v: vec3<f32>) -> f32 {
    return min(v.x, min(v.y, v.z));
}

fn wire_lerp(bary: vec3<f32>, object_edge_dist: vec3<f32>) -> f32 {
    let thickness = max(mat._Thickness, 0.0);
    if (kw(mat._SCREENSPACE)) {
        let dd = max(fwidth(bary), vec3<f32>(1.0e-5));
        let edge_px = min3(bary / dd);
        return 1.0 - smoothstep(
            max(thickness - 1.0, 0.0),
            thickness + 1.0,
            edge_px,
        );
    }

    let edge_dist = min3(object_edge_dist);
    let aa = max(fwidth(edge_dist), 1.0e-5);
    return 1.0 - smoothstep(
        max(thickness - aa, 0.0),
        thickness + aa,
        edge_dist,
    );
}

fn fresnel_amount(world_pos: vec3<f32>, world_n: vec3<f32>) -> f32 {
    let view_dir = normalize(rg::frame.camera_world_pos.xyz - world_pos);
    return pow(
        max(1.0 - abs(dot(normalize(world_n), view_dir)), 0.0),
        max(mat._Exp, 1.0e-4),
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
    @location(3) bary_in: vec4<f32>,
    @location(4) edge_dist_in: vec4<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = d.model * vec4<f32>(pos.xyz, 1.0);
    let world_n = normalize(d.normal_matrix * n.xyz);
    let model_scale = max_component_scale(d.model);
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
    out.world_pos = world_p.xyz;
    out.world_n = world_n;
    out.uv0 = uvu::apply_st(uv0, mat._MainTex_ST);
    out.bary = bary_in.xyz;
    out.object_edge_dist = edge_dist_in.xyz * model_scale;
    return out;
}

//#material alpha_blend
@fragment
fn fs_main(
    @builtin(front_facing) front_facing: bool,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) bary: vec3<f32>,
    @location(4) object_edge_dist: vec3<f32>,
) -> @location(0) vec4<f32> {
    let l = wire_lerp(bary, object_edge_dist);
    let tex = ts::sample_tex_2d(_MainTex, _MainTex_sampler, uv0, mat._MainTex_LodBias);

    var fill_color = select(mat._InnerFillColor, mat._FillColor, front_facing);
    var line_color = select(mat._InnerLineColor, mat._LineColor, front_facing);

    if (kw(mat._FRESNEL)) {
        let fresnel = fresnel_amount(world_pos, world_n);
        let fill_far = select(mat._InnerFillFarColor, mat._FillFarColor, front_facing);
        let line_far = select(mat._InnerLineFarColor, mat._LineFarColor, front_facing);
        fill_color = mix(fill_color, fill_far, fresnel);
        line_color = mix(line_color, line_far, fresnel);
    }

    let color = mix(fill_color, line_color, l) * tex;
    return rg::retain_globals_additive(color);
}
