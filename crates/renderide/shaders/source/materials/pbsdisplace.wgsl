//! Unity displaced metallic PBS (`Shader "PBSDisplace"`).
//!
//! This mirrors the base shader's opaque Standard forward path with optional vertex displacement,
//! UV offset, alpha clip, and position-driven displacement UVs.
//!
//! Renderide does not receive Unity shader keyword variants directly. `VERTEX_OFFSET` and
//! `UV_OFFSET` are inferred from their bound textures. `OBJECT_POS_OFFSET` becomes the default
//! position-offset mode when `_PositionOffsetMap` is bound unless the host explicitly signals
//! `VERTEX_POS_OFFSET`.

// unity-shader-name: PBSDisplace

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::cluster as pcls
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd
#import renderide::texture_sampling as ts

struct PbsDisplaceMaterial {
    _Color: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _PositionOffsetMagnitude: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _VertexOffsetMap_ST: vec4<f32>,
    _UVOffsetMap_ST: vec4<f32>,
    _PositionOffsetMap_ST: vec4<f32>,
    _AlphaClip: f32,
    _NormalScale: f32,
    _Glossiness: f32,
    _Metallic: f32,
    _VertexOffsetMagnitude: f32,
    _VertexOffsetBias: f32,
    _UVOffsetMagnitude: f32,
    _UVOffsetBias: f32,
    _ALBEDOTEX: f32,
    _EMISSIONTEX: f32,
    _NORMALMAP: f32,
    _METALLICMAP: f32,
    _OCCLUSION: f32,
    _ALPHACLIP: f32,
    VERTEX_OFFSET: f32,
    UV_OFFSET: f32,
    OBJECT_POS_OFFSET: f32,
    VERTEX_POS_OFFSET: f32,
    _MainTex_LodBias: f32,
    _NormalMap_LodBias: f32,
    _EmissionMap_LodBias: f32,
    _MetallicMap_LodBias: f32,
    _OcclusionMap_LodBias: f32,
    _VertexOffsetMap_LodBias: f32,
    _UVOffsetMap_LodBias: f32,
    _PositionOffsetMap_LodBias: f32,
}

@group(1) @binding(0)  var<uniform> mat: PbsDisplaceMaterial;
@group(1) @binding(1)  var _MainTex: texture_2d<f32>;
@group(1) @binding(2)  var _MainTex_sampler: sampler;
@group(1) @binding(3)  var _NormalMap: texture_2d<f32>;
@group(1) @binding(4)  var _NormalMap_sampler: sampler;
@group(1) @binding(5)  var _EmissionMap: texture_2d<f32>;
@group(1) @binding(6)  var _EmissionMap_sampler: sampler;
@group(1) @binding(7)  var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(8)  var _OcclusionMap_sampler: sampler;
@group(1) @binding(9)  var _MetallicMap: texture_2d<f32>;
@group(1) @binding(10) var _MetallicMap_sampler: sampler;
@group(1) @binding(11) var _VertexOffsetMap: texture_2d<f32>;
@group(1) @binding(12) var _VertexOffsetMap_sampler: sampler;
@group(1) @binding(13) var _UVOffsetMap: texture_2d<f32>;
@group(1) @binding(14) var _UVOffsetMap_sampler: sampler;
@group(1) @binding(15) var _PositionOffsetMap: texture_2d<f32>;
@group(1) @binding(16) var _PositionOffsetMap_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
}

struct SurfaceData {
    base_color: vec3<f32>,
    alpha: f32,
    metallic: f32,
    roughness: f32,
    occlusion: f32,
    normal: vec3<f32>,
    emission: vec3<f32>,
}

fn apply_st(uv_in: vec2<f32>, st: vec4<f32>) -> vec2<f32> {
    let uv = uv_in * st.xy + st.zw;
    return vec2<f32>(uv.x, 1.0 - uv.y);
}

fn vertex_offset_enabled() -> bool {
    return uvu::kw_enabled(mat.VERTEX_OFFSET);
}

fn uv_offset_enabled() -> bool {
    return uvu::kw_enabled(mat.UV_OFFSET);
}

fn use_vertex_pos_offset() -> bool {
    return uvu::kw_enabled(mat.VERTEX_POS_OFFSET);
}

fn use_object_pos_offset() -> bool {
    return uvu::kw_enabled(mat.OBJECT_POS_OFFSET) && !use_vertex_pos_offset();
}

fn position_offset_sample_uv(world_translation_xz: vec2<f32>, vertex_world_xz: vec2<f32>) -> vec2<f32> {
    let pos_driver = select(world_translation_xz, vertex_world_xz, use_vertex_pos_offset());
    return apply_st(pos_driver, mat._PositionOffsetMap_ST);
}

fn displaced_local_position(
    local_pos: vec3<f32>,
    local_n: vec3<f32>,
    world_translation_xz: vec2<f32>,
    vertex_world_xz: vec2<f32>,
    uv0: vec2<f32>,
) -> vec3<f32> {
    if (!vertex_offset_enabled()) {
        return local_pos;
    }

    var displacement_uv = uv0;
    if (use_object_pos_offset() || use_vertex_pos_offset()) {
        let pos_uv = position_offset_sample_uv(world_translation_xz, vertex_world_xz);
        let pos_offset = textureSampleLevel(
            _PositionOffsetMap,
            _PositionOffsetMap_sampler,
            pos_uv,
            0.0,
        ).xy * mat._PositionOffsetMagnitude.xy;
        displacement_uv = displacement_uv + pos_offset;
    }

    let offset_uv = apply_st(displacement_uv, mat._VertexOffsetMap_ST);
    let height = textureSampleLevel(
        _VertexOffsetMap,
        _VertexOffsetMap_sampler,
        offset_uv,
        0.0,
    ).r;
    let displacement = height * mat._VertexOffsetMagnitude + mat._VertexOffsetBias;
    return local_pos + local_n * displacement;
}

fn displaced_main_uv(uv0: vec2<f32>) -> vec2<f32> {
    var main_uv = uvu::apply_st(uv0, mat._MainTex_ST);
    if (uv_offset_enabled()) {
        let uv_offset_uv = apply_st(uv0, mat._UVOffsetMap_ST);
        let uv_offset = ts::sample_tex_2d(
            _UVOffsetMap,
            _UVOffsetMap_sampler,
            uv_offset_uv,
            mat._UVOffsetMap_LodBias,
        ).xy;
        main_uv = main_uv + uv_offset * mat._UVOffsetMagnitude + vec2<f32>(mat._UVOffsetBias);
    }
    return main_uv;
}

fn sample_normal_world(uv_main: vec2<f32>, world_n: vec3<f32>, front_facing: bool) -> vec3<f32> {
    let tbn = brdf::orthonormal_tbn(normalize(world_n));
    var ts_n = vec3<f32>(0.0, 0.0, 1.0);
    if (uvu::kw_enabled(mat._NORMALMAP)) {
        ts_n = nd::decode_ts_normal_with_placeholder_sample(
            ts::sample_tex_2d(_NormalMap, _NormalMap_sampler, uv_main, mat._NormalMap_LodBias),
            mat._NormalScale,
        );
    }
    if (!front_facing) {
        ts_n = vec3<f32>(ts_n.x, ts_n.y, -ts_n.z);
    }
    return normalize(tbn * ts_n);
}

fn sample_surface(uv0: vec2<f32>, world_n: vec3<f32>, front_facing: bool) -> SurfaceData {
    let uv_main = displaced_main_uv(uv0);

    var c = mat._Color;
    if (uvu::kw_enabled(mat._ALBEDOTEX)) {
        c = ts::sample_tex_2d(_MainTex, _MainTex_sampler, uv_main, mat._MainTex_LodBias) * c;
    }

    if (uvu::kw_enabled(mat._ALPHACLIP) && c.a <= mat._AlphaClip) {
        discard;
    }

    var occlusion = 1.0;
    if (uvu::kw_enabled(mat._OCCLUSION)) {
        occlusion = ts::sample_tex_2d(
            _OcclusionMap,
            _OcclusionMap_sampler,
            uv_main,
            mat._OcclusionMap_LodBias,
        ).r;
    }

    var metallic = mat._Metallic;
    var smoothness = mat._Glossiness;
    if (uvu::kw_enabled(mat._METALLICMAP)) {
        let m = ts::sample_tex_2d(
            _MetallicMap,
            _MetallicMap_sampler,
            uv_main,
            mat._MetallicMap_LodBias,
        );
        metallic = m.r;
        smoothness = m.a;
    }
    metallic = clamp(metallic, 0.0, 1.0);
    smoothness = clamp(smoothness, 0.0, 1.0);
    let roughness = clamp(1.0 - smoothness, 0.045, 1.0);

    var emission = mat._EmissionColor.rgb;
    if (uvu::kw_enabled(mat._EMISSIONTEX)) {
        emission = emission * ts::sample_tex_2d(
            _EmissionMap,
            _EmissionMap_sampler,
            uv_main,
            mat._EmissionMap_LodBias,
        ).rgb;
    }

    return SurfaceData(
        c.rgb,
        c.a,
        metallic,
        roughness,
        occlusion,
        sample_normal_world(uv_main, world_n, front_facing),
        emission,
    );
}

fn clustered_direct_lighting(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    view_layer: u32,
    s: SurfaceData,
    include_directional: bool,
    include_local: bool,
) -> vec3<f32> {
    let cam = rg::frame.camera_world_pos.xyz;
    let v = normalize(cam - world_pos);
    let f0 = mix(vec3<f32>(0.04, 0.04, 0.04), s.base_color, s.metallic);

    let cluster_id = pcls::cluster_id_from_frag(
        frag_xy,
        world_pos,
        rg::frame.view_space_z_coeffs,
        rg::frame.view_space_z_coeffs_right,
        view_layer,
        rg::frame.viewport_width,
        rg::frame.viewport_height,
        rg::frame.cluster_count_x,
        rg::frame.cluster_count_y,
        rg::frame.cluster_count_z,
        rg::frame.near_clip,
        rg::frame.far_clip,
    );

    let count = rg::cluster_light_counts[cluster_id];
    let i_max = min(count, pcls::MAX_LIGHTS_PER_TILE);
    var lo = vec3<f32>(0.0, 0.0, 0.0);

    for (var i = 0u; i < i_max; i++) {
        let li = pcls::cluster_light_index_at(cluster_id, i);
        if (li >= rg::frame.light_count) {
            continue;
        }
        let light = rg::lights[li];
        let is_directional = light.light_type == 1u;
        if ((is_directional && !include_directional) || (!is_directional && !include_local)) {
            continue;
        }
        lo = lo + brdf::direct_radiance_metallic(
            light,
            world_pos,
            s.normal,
            v,
            s.roughness,
            s.metallic,
            s.base_color,
            f0,
        );
    }

    return lo * s.occlusion;
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
    let local_pos = displaced_local_position(
        pos.xyz,
        normalize(n.xyz),
        d.model[3].xz,
        (d.model * vec4<f32>(pos.xyz, 1.0)).xz,
        uv0,
    );
    let world_p = d.model * vec4<f32>(local_pos, 1.0);
    let wn = normalize(d.normal_matrix * n.xyz);
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
    out.world_n = wn;
    out.uv0 = uv0;
#ifdef MULTIVIEW
    out.view_layer = view_idx;
#else
    out.view_layer = 0u;
#endif
    return out;
}

//#material forward_base
@fragment
fn fs_forward_base(
    @builtin(position) frag_pos: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let s = sample_surface(uv0, world_n, front_facing);
    let direct = clustered_direct_lighting(frag_pos.xy, world_pos, view_layer, s, true, false);
    let ambient = vec3<f32>(0.03, 0.03, 0.03) * s.base_color * s.occlusion;
    return vec4<f32>(ambient + direct + s.emission, s.alpha);
}

//#material forward_add
@fragment
fn fs_forward_delta(
    @builtin(position) frag_pos: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let s = sample_surface(uv0, world_n, front_facing);
    let direct = clustered_direct_lighting(frag_pos.xy, world_pos, view_layer, s, false, true);
    return vec4<f32>(direct, s.alpha);
}
