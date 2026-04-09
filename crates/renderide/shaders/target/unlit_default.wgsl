struct PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX {
    view_proj_left: mat4x4<f32>,
    view_proj_right: mat4x4<f32>,
    model: mat4x4<f32>,
    _pad: array<vec4<f32>, 4>,
}

struct FrameGlobalsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX {
    camera_world_pos: vec4<f32>,
    view_space_z_coeffs: vec4<f32>,
    cluster_count_x: u32,
    cluster_count_y: u32,
    cluster_count_z: u32,
    near_clip: f32,
    far_clip: f32,
    light_count: u32,
    viewport_width: u32,
    viewport_height: u32,
}

struct GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX {
    position: vec3<f32>,
    align_pad_vec3_pos: f32,
    direction: vec3<f32>,
    align_pad_vec3_dir: f32,
    color: vec3<f32>,
    intensity: f32,
    range: f32,
    spot_cos_half_angle: f32,
    light_type: u32,
    align_pad_before_shadow: u32,
    shadow_strength: f32,
    shadow_near_plane: f32,
    shadow_bias: f32,
    shadow_normal_bias: f32,
    shadow_type: u32,
    align_pad_vec3_tail: vec3<u32>,
}

struct UnlitMaterial {
    _Color: vec4<f32>,
    _Tex_ST: vec4<f32>,
    _MaskTex_ST: vec4<f32>,
    _OffsetTex_ST: vec4<f32>,
    _OffsetMagnitude: vec4<f32>,
    _Cutoff: f32,
    _PolarPow: f32,
    flags: u32,
    _SrcBlend: f32,
    _DstBlend: f32,
    _ZWrite: f32,
    _Cull: f32,
    _ZTest: f32,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(2) @binding(0) 
var<uniform> drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX: PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX;
@group(0) @binding(0) 
var<uniform> frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: FrameGlobalsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX;
@group(0) @binding(1) 
var<storage> lightsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: array<GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX>;
@group(0) @binding(2) 
var<storage> cluster_light_countsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: array<u32>;
@group(0) @binding(3) 
var<storage> cluster_light_indicesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: array<u32>;
@group(1) @binding(0) 
var<uniform> mat: UnlitMaterial;
@group(1) @binding(1) 
var _Tex: texture_2d<f32>;
@group(1) @binding(2) 
var _Tex_sampler: sampler;
@group(1) @binding(3) 
var _OffsetTex: texture_2d<f32>;
@group(1) @binding(4) 
var _OffsetTex_sampler: sampler;
@group(1) @binding(5) 
var _MaskTex: texture_2d<f32>;
@group(1) @binding(6) 
var _MaskTex_sampler: sampler;

fn apply_st(uv_1: vec2<f32>, st: vec4<f32>) -> vec2<f32> {
    let uv_st: vec2<f32> = ((uv_1 * st.xy) + st.zw);
    return vec2<f32>(uv_st.x, (1f - uv_st.y));
}

@vertex 
fn vs_main(@location(0) pos: vec4<f32>, @location(1) _n: vec4<f32>, @location(2) uv: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;

    let _e3: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.model;
    let world_p: vec4<f32> = (_e3 * vec4<f32>(pos.xyz, 1f));
    let vp: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.view_proj_left;
    out.clip_pos = (vp * world_p);
    out.uv = uv;
    let _e16: VertexOutput = out;
    return _e16;
}

@fragment 
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var albedo: vec4<f32>;
    var uv_main: vec2<f32>;
    var local: bool;
    var lit: u32 = 0u;

    let _e3: vec4<f32> = mat._Color;
    albedo = _e3;
    let _e7: u32 = mat.flags;
    if ((_e7 & 1u) != 0u) {
        let _e16: vec4<f32> = mat._Tex_ST;
        let _e17: vec2<f32> = apply_st(in.uv, _e16);
        uv_main = _e17;
        let _e21: u32 = mat.flags;
        if ((_e21 & 4u) != 0u) {
            let _e29: vec4<f32> = mat._OffsetTex_ST;
            let _e30: vec2<f32> = apply_st(in.uv, _e29);
            let offset_s: vec4<f32> = textureSample(_OffsetTex, _OffsetTex_sampler, _e30);
            let _e34: vec2<f32> = uv_main;
            let _e38: vec4<f32> = mat._OffsetMagnitude;
            uv_main = (_e34 + (offset_s.xy * _e38.xy));
        }
        let _e44: vec2<f32> = uv_main;
        let t: vec4<f32> = textureSample(_Tex, _Tex_sampler, _e44);
        let _e46: vec4<f32> = albedo;
        albedo = (_e46 * t);
    }
    let _e50: u32 = mat.flags;
    if ((_e50 & 24u) != 0u) {
        let _e58: vec4<f32> = mat._MaskTex_ST;
        let _e59: vec2<f32> = apply_st(in.uv, _e58);
        let mask: vec4<f32> = textureSample(_MaskTex, _MaskTex_sampler, _e59);
        let mul: f32 = ((((mask.x + mask.y) + mask.z) * 0.33333334f) * mask.w);
        let _e74: u32 = mat.flags;
        if ((_e74 & 8u) != 0u) {
            let _e81: f32 = albedo.w;
            albedo.w = (_e81 * mul);
        }
        let _e85: u32 = mat.flags;
        if ((_e85 & 16u) != 0u) {
            let _e92: f32 = mat._Cutoff;
            if (mul <= _e92) {
                discard;
            }
        }
    }
    let _e96: u32 = mat.flags;
    if ((_e96 & 2u) != 0u) {
        let _e103: u32 = mat.flags;
        local = ((_e103 & 16u) == 0u);
    } else {
        local = false;
    }
    let _e111: bool = local;
    if _e111 {
        let _e113: f32 = albedo.w;
        let _e116: f32 = mat._Cutoff;
        if (_e113 <= _e116) {
            discard;
        }
    }
    let _e120: u32 = mat.flags;
    if ((_e120 & 32u) != 0u) {
        let _e125: vec4<f32> = albedo;
        let _e128: f32 = albedo.w;
        let _e131: f32 = albedo.w;
        albedo = vec4<f32>((_e125.xyz * _e128), _e131);
    }
    let _e135: u32 = mat.flags;
    if ((_e135 & 64u) != 0u) {
        let _e141: f32 = albedo.x;
        let _e143: f32 = albedo.y;
        let _e146: f32 = albedo.z;
        let lum: f32 = (((_e141 + _e143) + _e146) * 0.33333334f);
        let _e152: f32 = albedo.w;
        albedo.w = (_e152 * lum);
    }
    let _e156: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.light_count;
    if (_e156 > 0u) {
        let _e162: u32 = lightsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0].light_type;
        lit = _e162;
    }
    let _e166: u32 = cluster_light_countsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0];
    let _e174: u32 = cluster_light_indicesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0];
    let cluster_touch: f32 = ((f32((_e166 & 255u)) * 0.0000000001f) + (f32((_e174 & 255u)) * 0.0000000001f));
    let _e181: vec4<f32> = albedo;
    let _e182: u32 = lit;
    return (_e181 + vec4<f32>(vec3(((f32(_e182) * 0.0000000001f) + cluster_touch)), 0f));
}
