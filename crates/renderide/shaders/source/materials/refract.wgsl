//! Unity shader `Shader "Filters/Refract"`: transparent grab-pass refraction.
//!
//! Routed via the AssetBundle `m_Container` leaf stem `refract` (parent folders are stripped by
//! [`crate::assets::shader::unity_asset::shader_logical_name_from_container_asset_path`]), so this
//! file is named `refract.wgsl` even though the Unity shader is declared as `Filters/Refract`.
//!
//! Samples the scene-color snapshot bound at `@group(0) @binding(6/7)` (populated by
//! [`crate::backend::frame_gpu::FrameGpuResources::copy_scene_color_snapshot`] before the
//! intersection subpass) at perturbed UVs derived from a tangent-space normal map. The `_GrabPass`
//! marker on the material uniform is what
//! [`crate::materials::wgsl_reflect::material_uniform_requires_grab_pass_subpass`] looks for to
//! schedule the snapshot copy.
//!
//! Reproduces the EVR `evrCalculateRefractionCoords` flow:
//!  1. Edge vignette `saturate((1 - |screen_uv*2-1|) * 32)` to fade refraction near the snapshot border.
//!  2. Depth fade `(scene_eye - surface_eye) / _DepthDivisor` so distant backgrounds get more shift.
//!  3. Perspective-correct screen-space offset `(view_n.xy / clip_w) * strength`.
//!  4. Occlusion safeguard: revert to the un-perturbed UV when the perturbed sample sits in front
//!     of the surface (else foreground geometry would bleed through the refractor).

// unity-shader-name: refract

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd

struct FiltersRefractMaterial {
    _NormalMap_ST: vec4<f32>,
    /// `(x, y, width, height)` in object space when `_RECTCLIP` is enabled (UI mask parity).
    _Rect: vec4<f32>,
    _RefractionStrength: f32,
    _DepthBias: f32,
    _DepthDivisor: f32,
    /// Marker field — its presence is what
    /// [`crate::materials::wgsl_reflect::material_uniform_requires_grab_pass_subpass`] keys on.
    _GrabPass: f32,
    _NORMALMAP: f32,
    _RECTCLIP: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(1) @binding(0) var<uniform> mat: FiltersRefractMaterial;
@group(1) @binding(1) var _NormalMap: texture_2d<f32>;
@group(1) @binding(2) var _NormalMap_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) obj_xy: vec2<f32>,
    @location(4) @interpolate(flat) cam_right: vec3<f32>,
    @location(5) @interpolate(flat) cam_up: vec3<f32>,
    @location(6) clip_w: f32,
    @location(7) @interpolate(flat) view_layer: u32,
}

// Math row-i of `view_proj` equals row-i of `P * V`. With axis-aligned projection
// (P diagonal in upper-left), row 0 of VP is `P[0][0] * camera_right_world` and row 1 is
// `P[1][1] * camera_up_world`, so normalizing recovers the world-space camera basis without
// needing a separate view-matrix uniform. WGSL stores `mat4x4` column-major, so math row-i
// is `(m[0][i], m[1][i], m[2][i])`.
fn camera_basis_from_vp(vp: mat4x4<f32>) -> mat3x3<f32> {
    let row0 = vec3<f32>(vp[0].x, vp[1].x, vp[2].x);
    let row1 = vec3<f32>(vp[0].y, vp[1].y, vp[2].y);
    let row2 = vec3<f32>(vp[0].z, vp[1].z, vp[2].z);
    return mat3x3<f32>(normalize(row0), normalize(row1), normalize(row2));
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
    let clip = vp * world_p;
    let basis = camera_basis_from_vp(vp);

    var out: VertexOutput;
    out.clip_pos = clip;
    out.world_pos = world_p.xyz;
    out.world_n = wn;
    out.uv0 = uv0;
    out.obj_xy = pos.xy;
    out.cam_right = basis[0];
    out.cam_up = basis[1];
    out.clip_w = clip.w;
#ifdef MULTIVIEW
    out.view_layer = view_idx;
#else
    out.view_layer = 0u;
#endif
    return out;
}

fn screen_uv_from_frag(frag_xy: vec2<f32>) -> vec2<f32> {
    let dims = max(
        vec2<f32>(f32(rg::frame.viewport_width), f32(rg::frame.viewport_height)),
        vec2<f32>(1.0, 1.0),
    );
    return frag_xy / dims;
}

fn fragment_eye_depth(world_pos: vec3<f32>, view_layer: u32) -> f32 {
    let z_coeffs = select(rg::frame.view_space_z_coeffs, rg::frame.view_space_z_coeffs_right, view_layer != 0u);
    let view_z = dot(z_coeffs.xyz, world_pos) + z_coeffs.w;
    return -view_z;
}

fn scene_eye_depth_at(uv: vec2<f32>, view_layer: u32) -> f32 {
    let dims = vec2<i32>(i32(rg::frame.viewport_width), i32(rg::frame.viewport_height));
    let max_xy = max(dims - vec2<i32>(1, 1), vec2<i32>(0, 0));
    let pixel = clamp(vec2<i32>(uv * vec2<f32>(dims)), vec2<i32>(0, 0), max_xy);
#ifdef MULTIVIEW
    let raw = textureLoad(rg::scene_depth_array, pixel, i32(view_layer), 0);
#else
    let raw = textureLoad(rg::scene_depth, pixel, 0);
#endif
    let denom = max(
        raw * (rg::frame.far_clip - rg::frame.near_clip) + rg::frame.near_clip,
        1e-6,
    );
    return (rg::frame.near_clip * rg::frame.far_clip) / denom;
}

fn sample_scene_color(uv: vec2<f32>, view_layer: u32) -> vec4<f32> {
#ifdef MULTIVIEW
    return textureSampleLevel(rg::scene_color_array, rg::scene_color_sampler, uv, i32(view_layer), 0.0);
#else
    return textureSampleLevel(rg::scene_color, rg::scene_color_sampler, uv, 0.0);
#endif
}

@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) obj_xy: vec2<f32>,
    @location(4) @interpolate(flat) cam_right: vec3<f32>,
    @location(5) @interpolate(flat) cam_up: vec3<f32>,
    @location(6) clip_w: f32,
    @location(7) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    if (uvu::kw_enabled(mat._RECTCLIP)) {
        let inside =
            obj_xy.x >= mat._Rect.x && obj_xy.x <= mat._Rect.x + mat._Rect.z &&
            obj_xy.y >= mat._Rect.y && obj_xy.y <= mat._Rect.y + mat._Rect.w;
        if (!inside) {
            discard;
        }
    }

    let screen_uv = screen_uv_from_frag(frag_pos.xy);
    let surface_eye = fragment_eye_depth(world_pos, view_layer);

    // Tangent-space normal mapping with a screen-aligned surface frame: Filters/Refract perturbs
    // in screen space rather than tracking artist-authored tangents, so projecting the camera
    // basis onto the surface plane reproduces the original asset's intent without requiring a
    // tangent vertex stream on every refractor mesh.
    var n = normalize(world_n);
    if (uvu::kw_enabled(mat._NORMALMAP)) {
        let t = normalize(cam_right - n * dot(cam_right, n));
        let b = normalize(cam_up - n * dot(cam_up, n));
        let raw = textureSample(_NormalMap, _NormalMap_sampler, uvu::apply_st(uv0, mat._NormalMap_ST));
        let ts_n = nd::decode_ts_normal_with_placeholder_sample(raw, 1.0);
        n = normalize(mat3x3<f32>(t, b, n) * ts_n);
    }

    let view_n_xy = vec2<f32>(dot(cam_right, n), dot(cam_up, n));

    let edge = saturate((vec2<f32>(1.0, 1.0) - abs(screen_uv * 2.0 - vec2<f32>(1.0, 1.0))) * 32.0);
    let div = select(mat._DepthDivisor, 1.0, abs(mat._DepthDivisor) < 1e-6);
    let depth_fade = saturate((scene_eye_depth_at(screen_uv, view_layer) - surface_eye) / div);
    let strength = mix(0.0, depth_fade, edge.x * edge.y);

    var grab_uv = screen_uv - (view_n_xy / max(clip_w, 1e-4)) * (mat._RefractionStrength * strength);

    let sampled_eye = scene_eye_depth_at(grab_uv, view_layer);
    if (sampled_eye < surface_eye - mat._DepthBias) {
        grab_uv = screen_uv;
    }

    let color = sample_scene_color(grab_uv, view_layer);
    return rg::retain_globals_additive(vec4<f32>(color.rgb, 1.0));
}
