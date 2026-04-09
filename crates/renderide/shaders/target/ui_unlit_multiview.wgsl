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
    @location(1) obj_xy: vec2<f32>,
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
fn vs_main(@builtin(view_index) view_idx: u32, @location(0) pos: vec4<f32>, @location(1) _n: vec4<f32>, @location(2) uv: vec2<f32>) -> VertexOutput {
    var vp: mat4x4<f32>;
    var out: VertexOutput;

    let _e3: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.model;
    let world_p: vec4<f32> = (_e3 * vec4<f32>(pos.xyz, 1f));
    if (view_idx == 0u) {
        let _e13: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.view_proj_left;
        vp = _e13;
    } else {
        let _e17: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.view_proj_right;
        vp = _e17;
    }
    let _e20: mat4x4<f32> = vp;
    out.clip_pos = (_e20 * world_p);
    out.uv = uv;
    out.obj_xy = pos.xy;
    let _e26: VertexOutput = out;
    return _e26;
}

@fragment 
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color: vec4<f32>;
    var local: bool;
    var local_1: bool;
    var local_2: bool;
    var lit: u32 = 0u;

    let _e3: vec4<f32> = mat._Tint;
    color = _e3;
    let _e7: u32 = mat.flags;
    if ((_e7 & 1u) != 0u) {
        let _e16: vec4<f32> = mat._MainTex_ST;
        let _e17: vec2<f32> = uv_with_st(in.uv, _e16);
        let t: vec4<f32> = textureSample(_MainTex, _MainTex_sampler, _e17);
        let _e21: vec4<f32> = color;
        color = (_e21 * t);
    }
    let _e25: u32 = mat.flags;
    if ((_e25 & 4u) != 0u) {
        let r: vec4<f32> = mat._Rect;
        let min_v: vec2<f32> = r.xy;
        let max_v: vec2<f32> = (r.xy + r.zw);
        if !((in.obj_xy.x < min_v.x)) {
            local = (in.obj_xy.x > max_v.x);
        } else {
            local = true;
        }
        let _e49: bool = local;
        if !(_e49) {
            local_1 = (in.obj_xy.y < min_v.y);
        } else {
            local_1 = true;
        }
        let _e58: bool = local_1;
        if !(_e58) {
            local_2 = (in.obj_xy.y > max_v.y);
        } else {
            local_2 = true;
        }
        let _e67: bool = local_2;
        if _e67 {
            discard;
        }
    }
    let _e70: u32 = mat.flags;
    if ((_e70 & 48u) != 0u) {
        let _e78: vec4<f32> = mat._MaskTex_ST;
        let _e79: vec2<f32> = uv_with_st(in.uv, _e78);
        let mask: vec4<f32> = textureSample(_MaskTex, _MaskTex_sampler, _e79);
        let mul: f32 = ((((mask.x + mask.y) + mask.z) * 0.33333334f) * mask.w);
        let _e94: u32 = mat.flags;
        if ((_e94 & 16u) != 0u) {
            let _e101: f32 = color.w;
            color.w = (_e101 * mul);
        }
        let _e105: u32 = mat.flags;
        if ((_e105 & 32u) != 0u) {
            let _e112: f32 = mat._Cutoff;
            if (mul < _e112) {
                discard;
            }
        }
    }
    let _e116: u32 = mat.flags;
    if ((_e116 & 2u) != 0u) {
        let _e122: f32 = color.w;
        let _e125: f32 = mat._Cutoff;
        if (_e122 < _e125) {
            discard;
        }
    }
    let _e129: u32 = mat.flags;
    if ((_e129 & 8u) != 0u) {
        let o: vec4<f32> = mat._OverlayTint;
        let _e137: vec4<f32> = color;
        let _e146: f32 = color.w;
        color = vec4<f32>((_e137.xyz * mix(vec3(1f), o.xyz, o.w)), _e146);
    }
    let _e150: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.light_count;
    if (_e150 > 0u) {
        let _e156: u32 = lightsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0].light_type;
        lit = _e156;
    }
    let _e160: u32 = cluster_light_countsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0];
    let _e168: u32 = cluster_light_indicesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0];
    let cluster_touch: f32 = ((f32((_e160 & 255u)) * 0.0000000001f) + (f32((_e168 & 255u)) * 0.0000000001f));
    let _e175: vec4<f32> = color;
    let _e176: u32 = lit;
    return (_e175 + vec4<f32>(vec3(((f32(_e176) * 0.0000000001f) + cluster_touch)), 0f));
}
