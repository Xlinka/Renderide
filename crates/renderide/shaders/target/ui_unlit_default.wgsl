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

struct UiUnlitMaterial {
    _MainTex_ST: vec4<f32>,
    _MaskTex_ST: vec4<f32>,
    _Tint: vec4<f32>,
    _OverlayTint: vec4<f32>,
    _Rect: vec4<f32>,
    _Cutoff: f32,
    _SrcBlend: f32,
    _DstBlend: f32,
    _ZWrite: f32,
    _Cull: f32,
    _StencilComp: f32,
    _Stencil: f32,
    _StencilOp: f32,
    _StencilWriteMask: f32,
    _StencilReadMask: f32,
    _ColorMask: f32,
    flags: u32,
    _pad_end: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) obj_xy: vec2<f32>,
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
var<uniform> mat: UiUnlitMaterial;
@group(1) @binding(1) 
var _MainTex: texture_2d<f32>;
@group(1) @binding(2) 
var _MainTex_sampler: sampler;
@group(1) @binding(3) 
var _MaskTex: texture_2d<f32>;
@group(1) @binding(4) 
var _MaskTex_sampler: sampler;

fn uv_with_st(uv_1: vec2<f32>, st: vec4<f32>) -> vec2<f32> {
    let uv_st: vec2<f32> = ((uv_1 * st.xy) + st.zw);
    return vec2<f32>(uv_st.x, (1f - uv_st.y));
}

@vertex 
fn vs_main(@location(0) pos: vec4<f32>, @location(1) _n: vec4<f32>, @location(2) uv: vec2<f32>, @location(3) color: vec4<f32>) -> VertexOutput {
    var out: VertexOutput;

    let _e3: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.model;
    let world_p: vec4<f32> = (_e3 * vec4<f32>(pos.xyz, 1f));
    let vp: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.view_proj_left;
    out.clip_pos = (vp * world_p);
    out.uv = uv;
    let _e20: vec4<f32> = mat._Tint;
    out.color = (color * _e20);
    out.obj_xy = pos.xy;
    let _e24: VertexOutput = out;
    return _e24;
}

@fragment 
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color_1: vec4<f32>;
    var local: bool;
    var local_1: bool;
    var local_2: bool;
    var local_3: bool;
    var lit: u32 = 0u;

    color_1 = in.color;
    let _e6: u32 = mat.flags;
    if ((_e6 & 1u) != 0u) {
        let _e14: vec4<f32> = mat._MainTex_ST;
        let _e15: vec2<f32> = uv_with_st(in.uv, _e14);
        let t: vec4<f32> = textureSample(_MainTex, _MainTex_sampler, _e15);
        let _e19: vec4<f32> = color_1;
        color_1 = (_e19 * t);
    }
    let _e23: u32 = mat.flags;
    if ((_e23 & 4u) != 0u) {
        let r: vec4<f32> = mat._Rect;
        let min_v: vec2<f32> = r.xy;
        let max_v: vec2<f32> = (r.xy + r.zw);
        if !((in.obj_xy.x < min_v.x)) {
            local = (in.obj_xy.x > max_v.x);
        } else {
            local = true;
        }
        let _e47: bool = local;
        if !(_e47) {
            local_1 = (in.obj_xy.y < min_v.y);
        } else {
            local_1 = true;
        }
        let _e56: bool = local_1;
        if !(_e56) {
            local_2 = (in.obj_xy.y > max_v.y);
        } else {
            local_2 = true;
        }
        let _e65: bool = local_2;
        if _e65 {
            discard;
        }
    }
    let _e68: u32 = mat.flags;
    if ((_e68 & 48u) != 0u) {
        let _e76: vec4<f32> = mat._MaskTex_ST;
        let _e77: vec2<f32> = uv_with_st(in.uv, _e76);
        let mask: vec4<f32> = textureSample(_MaskTex, _MaskTex_sampler, _e77);
        let mul: f32 = ((((mask.x + mask.y) + mask.z) * 0.33333334f) * mask.w);
        let _e92: u32 = mat.flags;
        if ((_e92 & 16u) != 0u) {
            let _e99: f32 = color_1.w;
            color_1.w = (_e99 * mul);
        }
        let _e103: u32 = mat.flags;
        if ((_e103 & 32u) != 0u) {
            let _e110: f32 = mat._Cutoff;
            if (mul <= _e110) {
                discard;
            }
        }
    }
    let _e114: u32 = mat.flags;
    if ((_e114 & 2u) != 0u) {
        let _e121: u32 = mat.flags;
        local_3 = ((_e121 & 32u) == 0u);
    } else {
        local_3 = false;
    }
    let _e129: bool = local_3;
    if _e129 {
        let _e131: f32 = color_1.w;
        let _e134: f32 = mat._Cutoff;
        if (_e131 <= _e134) {
            discard;
        }
    }
    let _e138: u32 = mat.flags;
    if ((_e138 & 8u) != 0u) {
        let o: vec4<f32> = mat._OverlayTint;
        let _e146: vec4<f32> = color_1;
        let _e155: f32 = color_1.w;
        color_1 = vec4<f32>((_e146.xyz * mix(vec3(1f), o.xyz, o.w)), _e155);
    }
    let _e159: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.light_count;
    if (_e159 > 0u) {
        let _e165: u32 = lightsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0].light_type;
        lit = _e165;
    }
    let _e169: u32 = cluster_light_countsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0];
    let _e177: u32 = cluster_light_indicesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0];
    let cluster_touch: f32 = ((f32((_e169 & 255u)) * 0.0000000001f) + (f32((_e177 & 255u)) * 0.0000000001f));
    let _e184: vec4<f32> = color_1;
    let _e185: u32 = lit;
    return (_e184 + vec4<f32>(vec3(((f32(_e185) * 0.0000000001f) + cluster_touch)), 0f));
}
