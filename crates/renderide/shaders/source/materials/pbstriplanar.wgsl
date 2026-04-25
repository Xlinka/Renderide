//! Unity triplanar opaque PBS (`Shader "PBSTriplanar"`).
//!
//! This follows the renderer's clustered forward metallic path and reconstructs the shader's
//! triplanar albedo / normal / metallic / emission / occlusion sampling in WGSL.
//!
//! The original Unity shader toggles `_WORLDSPACE` vs `_OBJECTSPACE` through shader keywords.
//! Renderide does not receive shader keyword variants over IPC, so world-space projection is the
//! fallback when neither keyword field is populated. If the host ever exposes `_OBJECTSPACE` as a
//! numeric material field, this shader will respect it.

// unity-shader-name: PBSTriplanar

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::cluster as pcls
#import renderide::normal_decode as nd
#import renderide::texture_sampling as ts

struct PbsTriplanarMaterial {
    _Color: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _NormalScale: f32,
    _Glossiness: f32,
    _Metallic: f32,
    _TriBlendPower: f32,
    _WORLDSPACE: f32,
    _OBJECTSPACE: f32,
    _ALBEDOTEX: f32,
    _EMISSIONTEX: f32,
    _NORMALMAP: f32,
    _METALLICMAP: f32,
    _OCCLUSION: f32,
    _MainTex_LodBias: f32,
    _NormalMap_LodBias: f32,
    _EmissionMap_LodBias: f32,
    _MetallicMap_LodBias: f32,
    _OcclusionMap_LodBias: f32,
}

@group(1) @binding(0)  var<uniform> mat: PbsTriplanarMaterial;
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

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) local_pos: vec3<f32>,
    @location(3) local_n: vec3<f32>,
    @location(4) normal_m0: vec3<f32>,
    @location(5) normal_m1: vec3<f32>,
    @location(6) normal_m2: vec3<f32>,
    @location(7) @interpolate(flat) view_layer: u32,
}

struct SurfaceData {
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    occlusion: f32,
    normal: vec3<f32>,
    emission: vec3<f32>,
}

struct TriplanarProjection {
    blend: vec3<f32>,
    uv_x: vec2<f32>,
    uv_y: vec2<f32>,
    uv_z: vec2<f32>,
    axis_normal: vec3<f32>,
    axis_sign: vec3<f32>,
}

fn kw(v: f32) -> bool {
    return v > 0.5;
}

fn use_object_space() -> bool {
    return kw(mat._OBJECTSPACE);
}

fn face_oriented_normal(n: vec3<f32>, front_facing: bool) -> vec3<f32> {
    if (front_facing) {
        return normalize(n);
    }
    return normalize(-n);
}

fn apply_planar_st(uv_in: vec2<f32>) -> vec2<f32> {
    let uv = uv_in * mat._MainTex_ST.xy + mat._MainTex_ST.zw;
    return vec2<f32>(uv.x, 1.0 - uv.y);
}

fn blend_rnm(n1_in: vec3<f32>, n2_in: vec3<f32>) -> vec3<f32> {
    let n1 = vec3<f32>(n1_in.xy, n1_in.z + 1.0);
    let n2 = vec3<f32>(-n2_in.xy, n2_in.z);
    return n1 * dot(n1, n2) / max(n1.z, 1.0e-4) - n2;
}

fn build_projection(axis_pos: vec3<f32>, axis_normal: vec3<f32>) -> TriplanarProjection {
    let blend_power = max(mat._TriBlendPower, 1.0e-4);
    var triblend = pow(abs(axis_normal), vec3<f32>(blend_power));
    triblend = triblend / max(dot(triblend, vec3<f32>(1.0, 1.0, 1.0)), 1.0e-4);

    var uv_x = apply_planar_st(axis_pos.zy);
    var uv_y = apply_planar_st(axis_pos.xz);
    var uv_z = apply_planar_st(axis_pos.xy);

    let axis_sign = select(
        vec3<f32>(-1.0, -1.0, -1.0),
        vec3<f32>(1.0, 1.0, 1.0),
        axis_normal >= vec3<f32>(0.0, 0.0, 0.0),
    );

    uv_x = vec2<f32>(uv_x.x * axis_sign.x, uv_x.y);
    uv_y = vec2<f32>(uv_y.x * axis_sign.y, uv_y.y);
    uv_z = vec2<f32>(uv_z.x * -axis_sign.z, uv_z.y);

    return TriplanarProjection(triblend, uv_x, uv_y, uv_z, axis_normal, axis_sign);
}

fn sample_triplanar(
    tex: texture_2d<f32>,
    samp: sampler,
    proj: TriplanarProjection,
    lod_bias: f32,
) -> vec4<f32> {
    let sx = ts::sample_tex_2d(tex, samp, proj.uv_x, lod_bias);
    let sy = ts::sample_tex_2d(tex, samp, proj.uv_y, lod_bias);
    let sz = ts::sample_tex_2d(tex, samp, proj.uv_z, lod_bias);
    return sx * proj.blend.x + sy * proj.blend.y + sz * proj.blend.z;
}

fn sample_triplanar_normal_world(
    proj: TriplanarProjection,
    world_base_normal: vec3<f32>,
    object_space: bool,
    normal_matrix: mat3x3<f32>,
) -> vec3<f32> {
    if (!kw(mat._NORMALMAP)) {
        return world_base_normal;
    }

    var tnormal_x = nd::decode_ts_normal_with_placeholder_sample(
        ts::sample_tex_2d(_NormalMap, _NormalMap_sampler, proj.uv_x, mat._NormalMap_LodBias),
        mat._NormalScale,
    );
    var tnormal_y = nd::decode_ts_normal_with_placeholder_sample(
        ts::sample_tex_2d(_NormalMap, _NormalMap_sampler, proj.uv_y, mat._NormalMap_LodBias),
        mat._NormalScale,
    );
    var tnormal_z = nd::decode_ts_normal_with_placeholder_sample(
        ts::sample_tex_2d(_NormalMap, _NormalMap_sampler, proj.uv_z, mat._NormalMap_LodBias),
        mat._NormalScale,
    );

    tnormal_x = vec3<f32>(tnormal_x.x * proj.axis_sign.x, tnormal_x.y, tnormal_x.z);
    tnormal_y = vec3<f32>(tnormal_y.x * proj.axis_sign.y, tnormal_y.y, tnormal_y.z);
    tnormal_z = vec3<f32>(tnormal_z.x * -proj.axis_sign.z, tnormal_z.y, tnormal_z.z);

    let abs_axis_normal = abs(proj.axis_normal);
    tnormal_x = blend_rnm(vec3<f32>(proj.axis_normal.zy, abs_axis_normal.x), tnormal_x);
    tnormal_y = blend_rnm(vec3<f32>(proj.axis_normal.xz, abs_axis_normal.y), tnormal_y);
    tnormal_z = blend_rnm(vec3<f32>(proj.axis_normal.xy, abs_axis_normal.z), tnormal_z);

    tnormal_x = vec3<f32>(tnormal_x.xy, tnormal_x.z * proj.axis_sign.x);
    tnormal_y = vec3<f32>(tnormal_y.xy, tnormal_y.z * proj.axis_sign.y);
    tnormal_z = vec3<f32>(tnormal_z.xy, tnormal_z.z * proj.axis_sign.z);

    let axis_space_normal = normalize(
        tnormal_x.zyx * proj.blend.x
            + tnormal_y.xzy * proj.blend.y
            + tnormal_z.xyz * proj.blend.z,
    );

    if (object_space) {
        return normalize(normal_matrix * axis_space_normal);
    }
    return axis_space_normal;
}

fn sample_surface(
    world_pos: vec3<f32>,
    world_n: vec3<f32>,
    local_pos: vec3<f32>,
    local_n: vec3<f32>,
    normal_matrix: mat3x3<f32>,
    front_facing: bool,
) -> SurfaceData {
    let world_base_normal = face_oriented_normal(world_n, front_facing);
    let local_base_normal = face_oriented_normal(local_n, front_facing);
    let object_space = use_object_space();

    var axis_pos = world_pos;
    var axis_normal = world_base_normal;
    if (object_space) {
        axis_pos = local_pos;
        axis_normal = local_base_normal;
    }
    let proj = build_projection(axis_pos, axis_normal);

    var base_color = mat._Color.rgb;
    if (kw(mat._ALBEDOTEX)) {
        base_color = sample_triplanar(_MainTex, _MainTex_sampler, proj, mat._MainTex_LodBias).rgb
            * base_color;
    }

    var metallic = mat._Metallic;
    var smoothness = mat._Glossiness;
    if (kw(mat._METALLICMAP)) {
        let metal = sample_triplanar(
            _MetallicMap,
            _MetallicMap_sampler,
            proj,
            mat._MetallicMap_LodBias,
        );
        metallic = metal.r;
        smoothness = metal.a;
    }
    metallic = clamp(metallic, 0.0, 1.0);
    smoothness = clamp(smoothness, 0.0, 1.0);
    let roughness = clamp(1.0 - smoothness, 0.045, 1.0);

    var emission = mat._EmissionColor.rgb;
    if (kw(mat._EMISSIONTEX)) {
        emission = emission
            * sample_triplanar(
                _EmissionMap,
                _EmissionMap_sampler,
                proj,
                mat._EmissionMap_LodBias,
            ).rgb;
    }

    var occlusion = 1.0;
    if (kw(mat._OCCLUSION)) {
        occlusion = sample_triplanar(
            _OcclusionMap,
            _OcclusionMap_sampler,
            proj,
            mat._OcclusionMap_LodBias,
        ).g;
    }

    let normal = sample_triplanar_normal_world(
        proj,
        world_base_normal,
        object_space,
        normal_matrix,
    );

    return SurfaceData(base_color, metallic, roughness, occlusion, normal, emission);
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
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = d.model * vec4<f32>(pos.xyz, 1.0);
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
    out.local_pos = pos.xyz;
    out.local_n = normalize(n.xyz);
    out.normal_m0 = d.normal_matrix[0];
    out.normal_m1 = d.normal_matrix[1];
    out.normal_m2 = d.normal_matrix[2];
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
    @location(2) local_pos: vec3<f32>,
    @location(3) local_n: vec3<f32>,
    @location(4) normal_m0: vec3<f32>,
    @location(5) normal_m1: vec3<f32>,
    @location(6) normal_m2: vec3<f32>,
    @location(7) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let s = sample_surface(
        world_pos,
        world_n,
        local_pos,
        local_n,
        mat3x3<f32>(normal_m0, normal_m1, normal_m2),
        front_facing,
    );
    let direct = clustered_direct_lighting(frag_pos.xy, world_pos, view_layer, s, true, false);
    let ambient = vec3<f32>(0.03, 0.03, 0.03) * s.base_color * s.occlusion;
    return vec4<f32>(ambient + direct + s.emission, 1.0);
}

//#material forward_add
@fragment
fn fs_forward_delta(
    @builtin(position) frag_pos: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) local_pos: vec3<f32>,
    @location(3) local_n: vec3<f32>,
    @location(4) normal_m0: vec3<f32>,
    @location(5) normal_m1: vec3<f32>,
    @location(6) normal_m2: vec3<f32>,
    @location(7) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let s = sample_surface(
        world_pos,
        world_n,
        local_pos,
        local_n,
        mat3x3<f32>(normal_m0, normal_m1, normal_m2),
        front_facing,
    );
    let direct = clustered_direct_lighting(frag_pos.xy, world_pos, view_layer, s, false, true);
    return vec4<f32>(direct, 0.0);
}
