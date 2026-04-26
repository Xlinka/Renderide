//! Unity shader `Shader "Volume/FogBox"`: volumetric fog inside an axis-aligned unit cube.
//!
//! Routed via the AssetBundle `m_Container` leaf stem `fogboxvolume` (the `Volume/` prefix is
//! stripped by [`crate::assets::shader::unity_asset::shader_logical_name_from_container_asset_path`]).
//!
//! Renders the **back faces** of a unit cube (`Cull Front`, `ZTest Always`, `ZWrite Off`) so the
//! effect appears even when the camera is inside the volume. The fragment shader walks the view
//! ray from the camera through the cube, clamps the ray's far end against the scene-depth
//! snapshot (`@group(0) @binding(4)`), and accumulates fog over the resulting segment length.
//!
//! ## Keyword permutations (runtime uniform flags via `kw_enabled`)
//!
//! - `_OBJECT_SPACE` (vs `_WORLD_SPACE`): ray math runs in object space (vs world space). Object
//!   space avoids the inverse-model multiply for `_WorldSpaceCameraPos` but loses the spherical
//!   approximation when the host applies non-uniform scale. Both modes use `affine_inverse(model)`
//!   to remap the camera into the unit cube.
//! - `_FOG_LINEAR` / `_FOG_EXP` / `_FOG_EXP2`: distance falloff curve. Linear clamps to
//!   `[_FogStart, _FogEnd]`; EXP uses `1 - 1/exp(d * density)`; EXP2 uses `1 - 1/exp((d*density)^2)`.
//! - `_COLOR_CONSTANT` (vs `_COLOR_VERT_GRADIENT`): single accumulation color vs Y-axis lerp
//!   between `_AccumulationColorBottom` (y=-0.5) and `_AccumulationColorTop` (y=+0.5).
//! - `_SATURATE_ALPHA` / `_SATURATE_COLOR`: clamp final alpha or full RGBA to `[0, 1]`.

// unity-shader-name: fogboxvolume

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::uv_utils as uvu

struct FogBoxVolumeMaterial {
    _BaseColor: vec4<f32>,
    _AccumulationColor: vec4<f32>,
    _AccumulationColorBottom: vec4<f32>,
    _AccumulationColorTop: vec4<f32>,
    _AccumulationRate: f32,
    _GammaCurve: f32,
    _FogStart: f32,
    _FogEnd: f32,
    _FogDensity: f32,
    _OBJECT_SPACE: f32,
    _FOG_LINEAR: f32,
    _FOG_EXP: f32,
    _FOG_EXP2: f32,
    _COLOR_CONSTANT: f32,
    _COLOR_VERT_GRADIENT: f32,
    _SATURATE_ALPHA: f32,
    _SATURATE_COLOR: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(1) @binding(0) var<uniform> mat: FogBoxVolumeMaterial;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) obj_pos: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) screen_uv_w: vec3<f32>,
    @location(3) eye_depth: f32,
    @location(4) @interpolate(flat) view_layer: u32,
}

// Inverse of a column-major mat3x3 via the cross-product identity. For an invertible matrix
// `[c0 c1 c2]`, `inverse = transpose([cross(c1,c2), cross(c2,c0), cross(c0,c1)] / det)` where
// `det = dot(c0, cross(c1, c2))`.
fn inverse_mat3(m: mat3x3<f32>) -> mat3x3<f32> {
    let c0 = m[0];
    let c1 = m[1];
    let c2 = m[2];
    let r0 = cross(c1, c2);
    let r1 = cross(c2, c0);
    let r2 = cross(c0, c1);
    let inv_det = 1.0 / max(dot(c0, r0), 1e-20);
    return mat3x3<f32>(
        vec3<f32>(r0.x, r1.x, r2.x) * inv_det,
        vec3<f32>(r0.y, r1.y, r2.y) * inv_det,
        vec3<f32>(r0.z, r1.z, r2.z) * inv_det,
    );
}

// Inverse of an affine `mat4x4` (rotation/scale/shear in upper 3x3, translation in column 3).
fn affine_inverse(m: mat4x4<f32>) -> mat4x4<f32> {
    let r3 = mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);
    let inv3 = inverse_mat3(r3);
    let inv_t = -(inv3 * m[3].xyz);
    return mat4x4<f32>(
        vec4<f32>(inv3[0], 0.0),
        vec4<f32>(inv3[1], 0.0),
        vec4<f32>(inv3[2], 0.0),
        vec4<f32>(inv_t, 1.0),
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

    // ComputeScreenPos equivalent (Y-flipped for WebGPU's V-down). The fragment recovers
    // sampling UVs as `screen_uv_w.xy / screen_uv_w.z`.
    let proj_xy = vec2<f32>(
        (clip.x + clip.w) * 0.5,
        (clip.w - clip.y) * 0.5,
    );

    // Eye-depth of this back-face vertex (positive in front of the camera).
    let z_coeffs = select(rg::frame.view_space_z_coeffs, rg::frame.view_space_z_coeffs_right,
#ifdef MULTIVIEW
        view_idx != 0u
#else
        false
#endif
    );
    let eye_depth = -(dot(z_coeffs.xyz, world_p.xyz) + z_coeffs.w);

    var out: VertexOutput;
    out.clip_pos = clip;
    out.obj_pos = pos.xyz;
    out.world_pos = world_p.xyz;
    out.screen_uv_w = vec3<f32>(proj_xy, max(clip.w, 1e-6));
    out.eye_depth = eye_depth;
#ifdef MULTIVIEW
    out.view_layer = view_idx;
#else
    out.view_layer = 0u;
#endif
    return out;
}

fn scene_eye_depth_at_pixel(pixel_xy: vec2<i32>, view_layer: u32) -> f32 {
    let dims = vec2<i32>(i32(rg::frame.viewport_width), i32(rg::frame.viewport_height));
    let max_xy = max(dims - vec2<i32>(1, 1), vec2<i32>(0, 0));
    let p = clamp(pixel_xy, vec2<i32>(0, 0), max_xy);
#ifdef MULTIVIEW
    let raw = textureLoad(rg::scene_depth_array, p, i32(view_layer), 0);
#else
    let raw = textureLoad(rg::scene_depth, p, 0);
#endif
    let denom = max(
        raw * (rg::frame.far_clip - rg::frame.near_clip) + rg::frame.near_clip,
        1e-6,
    );
    return (rg::frame.near_clip * rg::frame.far_clip) / denom;
}

// Line ↔ axis-aligned plane (`plane_pt + n * t`) intersection. Returns `line_pt` unchanged when
// the ray is parallel to the plane (`prod2 ≈ 0`); the caller filters those out via the bounds
// check in [`filter_point`].
fn line_plane_intersection(
    line_pt: vec3<f32>,
    line_dir: vec3<f32>,
    plane_pt: vec3<f32>,
    plane_n: vec3<f32>,
) -> vec3<f32> {
    let diff = line_pt - plane_pt;
    let prod1 = dot(diff, plane_n);
    let prod2 = dot(line_dir, plane_n);
    if (abs(prod2) < 1e-8) {
        return line_pt;
    }
    return line_pt - line_dir * (prod1 / prod2);
}

struct ClosestHit {
    point: vec3<f32>,
    dist: f32,
}

// Keep the candidate cube-face intersection only if both tangential coordinates lie inside the
// `[-0.5, 0.5]` face square, then track the nearest-to-`ref_pt` candidate.
fn filter_point(
    cur: ClosestHit,
    ref_pt: vec3<f32>,
    candidate: vec3<f32>,
    check_square: vec2<f32>,
) -> ClosestHit {
    if (any(abs(check_square) > vec2<f32>(0.5))) {
        return cur;
    }
    let d = distance(ref_pt, candidate);
    if (d < cur.dist) {
        return ClosestHit(candidate, d);
    }
    return cur;
}

// Find the closest entry/exit point on the unit cube along the line `(line_pt, line_dir)` by
// intersecting all six faces and keeping the nearest valid hit. Mirrors `IntersectUnitCube` in
// the Unity asset.
fn intersect_unit_cube(line_pt: vec3<f32>, line_dir: vec3<f32>) -> vec3<f32> {
    let i0 = line_plane_intersection(line_pt, line_dir, vec3<f32>(-0.5, 0.0, 0.0), vec3<f32>(-1.0, 0.0, 0.0));
    let i1 = line_plane_intersection(line_pt, line_dir, vec3<f32>(0.5, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0));
    let i2 = line_plane_intersection(line_pt, line_dir, vec3<f32>(0.0, -0.5, 0.0), vec3<f32>(0.0, -1.0, 0.0));
    let i3 = line_plane_intersection(line_pt, line_dir, vec3<f32>(0.0, 0.5, 0.0), vec3<f32>(0.0, 1.0, 0.0));
    let i4 = line_plane_intersection(line_pt, line_dir, vec3<f32>(0.0, 0.0, -0.5), vec3<f32>(0.0, 0.0, -1.0));
    let i5 = line_plane_intersection(line_pt, line_dir, vec3<f32>(0.0, 0.0, 0.5), vec3<f32>(0.0, 0.0, 1.0));

    var hit = ClosestHit(line_pt, 65000.0);
    hit = filter_point(hit, line_pt, i0, i0.yz);
    hit = filter_point(hit, line_pt, i1, i1.yz);
    hit = filter_point(hit, line_pt, i2, i2.xz);
    hit = filter_point(hit, line_pt, i3, i3.xz);
    hit = filter_point(hit, line_pt, i4, i4.xy);
    hit = filter_point(hit, line_pt, i5, i5.xy);
    return hit.point;
}

fn clamp_inside_unit_cube(p: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    if (all(abs(p) <= vec3<f32>(0.5, 0.5, 0.5))) {
        return p;
    }
    return intersect_unit_cube(p, dir);
}

fn distance_sqr(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let v = a - b;
    return dot(v, v);
}

fn fog_distance(raw: f32) -> f32 {
    if (uvu::kw_enabled(mat._FOG_LINEAR)) {
        var d = min(mat._FogEnd, raw);
        d = d - mat._FogStart;
        return max(0.0, d);
    }
    if (uvu::kw_enabled(mat._FOG_EXP)) {
        return 1.0 - (1.0 / exp(raw * mat._FogDensity));
    }
    if (uvu::kw_enabled(mat._FOG_EXP2)) {
        let dd = raw * mat._FogDensity;
        return 1.0 - (1.0 / exp(dd * dd));
    }
    // Default to linear without start/end (parity with FOG_LINEAR's degenerate `_FogStart=0`).
    return raw;
}

fn accumulation_color(start: vec3<f32>, end: vec3<f32>, world_to_object: mat4x4<f32>) -> vec4<f32> {
    if (uvu::kw_enabled(mat._COLOR_VERT_GRADIENT)) {
        var start_y: f32;
        var end_y: f32;
        if (uvu::kw_enabled(mat._OBJECT_SPACE)) {
            start_y = start.y;
            end_y = end.y;
        } else {
            let local_s = (world_to_object * vec4<f32>(start, 1.0)).xyz;
            let local_e = (world_to_object * vec4<f32>(end, 1.0)).xyz;
            let dir_s = normalize(select(local_s, vec3<f32>(0.0, 1.0, 0.0), all(abs(local_s) < vec3<f32>(1e-6))));
            let dir_e = normalize(select(local_e, vec3<f32>(0.0, 1.0, 0.0), all(abs(local_e) < vec3<f32>(1e-6))));
            start_y = clamp_inside_unit_cube(local_s, dir_s).y;
            end_y = clamp_inside_unit_cube(local_e, dir_e).y;
        }
        let avg_y = (start_y + end_y) * 0.5 + 0.5;
        return mix(mat._AccumulationColorBottom, mat._AccumulationColorTop, saturate(avg_y));
    }
    return mat._AccumulationColor;
}

//#material volume_fog
@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) obj_pos: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) screen_uv_w: vec3<f32>,
    @location(3) eye_depth: f32,
    @location(4) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    // Sample the scene-depth snapshot at the projective screen-UV (matching Unity's
    // `SAMPLE_DEPTH_TEXTURE_PROJ(_CameraDepthTexture, projPos)`).
    let proj_uv = screen_uv_w.xy / max(screen_uv_w.z, 1e-6);
    let dims = vec2<f32>(f32(rg::frame.viewport_width), f32(rg::frame.viewport_height));
    let scene_z = scene_eye_depth_at_pixel(vec2<i32>(proj_uv * dims), view_layer);
    let part_z = eye_depth;

    // Reconstruct world↔object transforms from the per-draw model matrix. Multiview shares the
    // same model so this is safe to read once.
    let model = pd::get_draw(0u).model;
    // (instance_index isn't available in fragments, but for a single-mesh fog box every draw call
    // routes through the same `model`. The vertex stage already used `pd::get_draw(instance_index)`
    // to project, and we only need the transform's *value* here, not per-instance variance.)
    let world_to_object = affine_inverse(model);
    let cam_world = rg::frame.camera_world_pos.xyz;

    var start: vec3<f32>;
    var end: vec3<f32>;
    var cam_pos_for_test: vec3<f32>;
    var dir: vec3<f32>;

    if (uvu::kw_enabled(mat._OBJECT_SPACE)) {
        let cam_obj = (world_to_object * vec4<f32>(cam_world, 1.0)).xyz;
        let end_obj = obj_pos;
        dir = normalize(end_obj - cam_obj);
        start = clamp_inside_unit_cube(cam_obj, dir);
        let max_dist = distance(cam_obj, end_obj);
        let end_ratio = min(scene_z / max(part_z, 1e-6), 1.0);
        end = cam_obj + dir * max_dist * end_ratio;
        cam_pos_for_test = cam_obj;
    } else {
        // World-space mode: clamp the camera into the unit cube along `-ndir` so we know the
        // ray's near end, then discard if the resulting clamped point sits behind opaque geometry.
        var clamped_obj = (world_to_object * vec4<f32>(cam_world, 1.0)).xyz;
        let world_dir = normalize(world_pos - cam_world);
        // The corresponding object-space direction (rotation only — translation is irrelevant for
        // a direction). This matches Unity's `i.origin.xyz - clampedStartPos` since `i.origin`
        // arrives in object space.
        let obj_n = normalize(obj_pos - clamped_obj);
        clamped_obj = clamp_inside_unit_cube(clamped_obj, -obj_n);

        // View-space depth of the clamped object-space starting point.
        let clamped_world = (model * vec4<f32>(clamped_obj, 1.0)).xyz;
        let z_coeffs = select(rg::frame.view_space_z_coeffs, rg::frame.view_space_z_coeffs_right, view_layer != 0u);
        let clamped_eye = -(dot(z_coeffs.xyz, clamped_world) + z_coeffs.w);
        if (clamped_eye > scene_z) {
            discard;
        }

        dir = world_dir;
        start = cam_world;
        end = cam_world + scene_z * dir;
        cam_pos_for_test = cam_world;
    }

    if (distance_sqr(cam_pos_for_test, end) < distance_sqr(cam_pos_for_test, start)) {
        discard;
    }

    let raw_dist = distance(start, end);
    let dist = fog_distance(raw_dist);

    let acc_color = accumulation_color(start, end, world_to_object);
    let acc = pow(max(dist * mat._AccumulationRate, 0.0), max(mat._GammaCurve, 1e-6)) * acc_color;
    var result = mat._BaseColor + acc;

    if (uvu::kw_enabled(mat._SATURATE_ALPHA)) {
        result.a = saturate(result.a);
    } else if (uvu::kw_enabled(mat._SATURATE_COLOR)) {
        result = saturate(result);
    }

    return rg::retain_globals_additive(result);
}
