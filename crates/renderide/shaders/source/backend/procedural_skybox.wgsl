//! Fullscreen procedural skybox ported from the base Unity `ProceduralSky` shader.
//! Drawn inside the main forward opaque pass before world meshes.

struct ProceduralSkyUniforms {
    view_to_world_left: mat4x4<f32>,
    view_to_world_right: mat4x4<f32>,
    inv_proj_left: mat4x4<f32>,
    inv_proj_right: mat4x4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    sky_tint: vec4<f32>,
    ground_color: vec4<f32>,
    /// `.x = sun_size`, `.y = atmosphere_thickness`, `.z = exposure`, `.w = sun_disk_mode`.
    params0: vec4<f32>,
}

@group(0) @binding(0) var<uniform> sky: ProceduralSkyUniforms;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var out: VsOut;
    out.clip_pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

const DEFAULT_SCATTERING_WAVELENGTH = vec3<f32>(0.65, 0.57, 0.475);
const SCATTERING_WAVELENGTH_RANGE = vec3<f32>(0.15, 0.15, 0.15);
const OUTER_RADIUS = 1.025;
const OUTER_RADIUS2 = OUTER_RADIUS * OUTER_RADIUS;
const INNER_RADIUS = 1.0;
const INNER_RADIUS2 = 1.0;
const CAMERA_HEIGHT = 0.0001;
const MIE = 0.001;
const SUN_BRIGHTNESS = 20.0;
const MAX_SCATTER = 50.0;
const SUN_SCALE = 400.0 * SUN_BRIGHTNESS;
const KM_E_SUN = MIE * SUN_BRIGHTNESS;
const KM_4_PI = MIE * 4.0 * 3.14159265;
const SCALE = 1.0 / (OUTER_RADIUS - 1.0);
const SCALE_DEPTH = 0.25;
const SCALE_OVER_SCALE_DEPTH = SCALE / SCALE_DEPTH;
const MIE_G = -0.990;
const MIE_G2 = 0.9801;
const SKY_GROUND_THRESHOLD = 0.02;

struct ScatteringSample {
    sky_color: vec3<f32>,
    ground_color: vec3<f32>,
    sun_color: vec3<f32>,
    sky_ground_factor: f32,
}

fn saturate1(v: f32) -> f32 {
    return clamp(v, 0.0, 1.0);
}

fn sun_size() -> f32 {
    return sky.params0.x;
}

fn atmosphere_thickness() -> f32 {
    return sky.params0.y;
}

fn exposure() -> f32 {
    return sky.params0.z;
}

fn sun_disk_mode() -> f32 {
    return sky.params0.w;
}

fn select_view_to_world(view_layer: u32) -> mat4x4<f32> {
#ifdef MULTIVIEW
    if (view_layer == 0u) {
        return sky.view_to_world_left;
    }
    return sky.view_to_world_right;
#else
    return sky.view_to_world_left;
#endif
}

fn select_inv_proj(view_layer: u32) -> mat4x4<f32> {
#ifdef MULTIVIEW
    if (view_layer == 0u) {
        return sky.inv_proj_left;
    }
    return sky.inv_proj_right;
#else
    return sky.inv_proj_left;
#endif
}

fn reconstruct_world_ray(uv: vec2<f32>, view_layer: u32) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let clip = vec4<f32>(ndc, 0.0, 1.0);
    let view_h = select_inv_proj(view_layer) * clip;
    let view_dir = normalize(view_h.xyz / max(abs(view_h.w), 1.0e-6));
    let world_h = select_view_to_world(view_layer) * vec4<f32>(view_dir, 0.0);
    return normalize(world_h.xyz);
}

fn scale(in_cos: f32) -> f32 {
    let x = 1.0 - in_cos;
    return 0.25 * exp(-0.00287 + x * (0.459 + x * (3.83 + x * (-6.80 + x * 5.25))));
}

fn get_rayleigh_phase(eye_cos2: f32) -> f32 {
    return 0.75 + 0.75 * eye_cos2;
}

fn get_rayleigh_phase_from_dirs(light: vec3<f32>, ray: vec3<f32>) -> f32 {
    let eye_cos = dot(light, ray);
    return get_rayleigh_phase(eye_cos * eye_cos);
}

fn get_mie_phase(eye_cos: f32, eye_cos2: f32) -> f32 {
    var temp = 1.0 + MIE_G2 - 2.0 * MIE_G * eye_cos;
    temp = pow(temp, pow(max(sun_size(), 0.0), 0.65) * 10.0);
    temp = max(temp, 1.0e-4);
    return 1.5 * ((1.0 - MIE_G2) / (2.0 + MIE_G2)) * (1.0 + eye_cos2) / temp;
}

fn calc_sun_spot(vec1: vec3<f32>, vec2: vec3<f32>) -> f32 {
    let delta = vec1 - vec2;
    let dist = length(delta);
    let radius = max(sun_size(), 1.0e-4);
    let spot = 1.0 - smoothstep(0.0, radius, dist);
    return SUN_SCALE * spot * spot;
}

fn compute_scattering(ray: vec3<f32>) -> ScatteringSample {
    let sky_tint_gamma = pow(max(sky.sky_tint.rgb, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
    let scattering_wavelength = mix(
        DEFAULT_SCATTERING_WAVELENGTH - SCATTERING_WAVELENGTH_RANGE,
        DEFAULT_SCATTERING_WAVELENGTH + SCATTERING_WAVELENGTH_RANGE,
        vec3<f32>(1.0) - sky_tint_gamma,
    );
    let inv_wavelength = 1.0 / pow(scattering_wavelength, vec3<f32>(4.0));

    let rayleigh = mix(0.0, 0.0025, pow(max(atmosphere_thickness(), 0.0), 2.5));
    let kr_e_sun = rayleigh * SUN_BRIGHTNESS;
    let kr_4_pi = rayleigh * 4.0 * 3.14159265;
    let camera_pos = vec3<f32>(0.0, INNER_RADIUS + CAMERA_HEIGHT, 0.0);
    let sun_direction = normalize(sky.sun_direction.xyz);

    var c_in = vec3<f32>(0.0);
    var c_out = vec3<f32>(0.0);

    if (ray.y >= 0.0) {
        let far_dist = sqrt(OUTER_RADIUS2 + INNER_RADIUS2 * ray.y * ray.y - INNER_RADIUS2)
            - INNER_RADIUS * ray.y;
        let height = INNER_RADIUS + CAMERA_HEIGHT;
        let depth = exp(SCALE_OVER_SCALE_DEPTH * (-CAMERA_HEIGHT));
        let start_angle = dot(ray, camera_pos) / height;
        let start_offset = depth * scale(start_angle);
        let sample_length = far_dist * 0.5;
        let scaled_length = sample_length * SCALE;
        let sample_ray = ray * sample_length;
        var sample_point = camera_pos + sample_ray * 0.5;
        var front_color = vec3<f32>(0.0);

        for (var i: i32 = 0; i < 2; i = i + 1) {
            let sample_height = length(sample_point);
            let sample_depth = exp(SCALE_OVER_SCALE_DEPTH * (INNER_RADIUS - sample_height));
            let light_angle = dot(sun_direction, sample_point) / sample_height;
            let camera_angle = dot(ray, sample_point) / sample_height;
            let scatter =
                start_offset + sample_depth * (scale(light_angle) - scale(camera_angle));
            let attenuate = exp(
                -clamp(scatter, 0.0, MAX_SCATTER) * (inv_wavelength * kr_4_pi + KM_4_PI),
            );
            front_color += attenuate * (sample_depth * scaled_length);
            sample_point += sample_ray;
        }

        c_in = front_color * (inv_wavelength * kr_e_sun);
        c_out = front_color * KM_E_SUN;
    } else {
        let far_dist = (-CAMERA_HEIGHT) / min(-0.001, ray.y);
        let pos = camera_pos + far_dist * ray;
        let depth = exp((-CAMERA_HEIGHT) * (1.0 / SCALE_DEPTH));
        let camera_angle = dot(-ray, pos);
        let light_angle = dot(sun_direction, pos);
        let camera_scale = scale(camera_angle);
        let light_scale = scale(light_angle);
        let camera_offset = depth * camera_scale;
        let temp = light_scale + camera_scale;
        let sample_length = far_dist * 0.5;
        let scaled_length = sample_length * SCALE;
        let sample_ray = ray * sample_length;
        var sample_point = camera_pos + sample_ray * 0.5;
        var front_color = vec3<f32>(0.0);
        var attenuate = vec3<f32>(0.0);

        for (var i: i32 = 0; i < 2; i = i + 1) {
            let sample_height = length(sample_point);
            let sample_depth = exp(SCALE_OVER_SCALE_DEPTH * (INNER_RADIUS - sample_height));
            let scatter = sample_depth * temp - camera_offset;
            attenuate = exp(
                -clamp(scatter, 0.0, MAX_SCATTER) * (inv_wavelength * kr_4_pi + KM_4_PI),
            );
            front_color += attenuate * (sample_depth * scaled_length);
            sample_point += sample_ray;
        }

        c_in = front_color * (inv_wavelength * kr_e_sun + KM_E_SUN);
        c_out = clamp(attenuate, vec3<f32>(0.0), vec3<f32>(1.0));
    }

    return ScatteringSample(
        exposure() * (c_in * get_rayleigh_phase_from_dirs(sun_direction, -ray)),
        exposure() * (c_in + sky.ground_color.rgb * c_out),
        exposure() * (c_out * sky.sun_color.rgb),
        -ray.y / SKY_GROUND_THRESHOLD,
    );
}

fn final_sky_color(ray: vec3<f32>) -> vec3<f32> {
    let scattering = compute_scattering(ray);
    let col = mix(
        scattering.sky_color,
        scattering.ground_color,
        vec3<f32>(saturate1(scattering.sky_ground_factor)),
    );
    if (scattering.sky_ground_factor >= 0.0 || sun_disk_mode() < 0.5) {
        return col;
    }

    let sun_direction = normalize(sky.sun_direction.xyz);
    if (sun_disk_mode() < 1.5) {
        let mie = calc_sun_spot(sun_direction, -ray);
        return col + mie * scattering.sun_color;
    }

    let eye_cos = dot(sun_direction, ray);
    let eye_cos2 = eye_cos * eye_cos;
    let mie = get_mie_phase(eye_cos, eye_cos2);
    return col + mie * scattering.sun_color;
}

#ifdef MULTIVIEW
@fragment
fn fs_main(in: VsOut, @builtin(view_index) view: u32) -> @location(0) vec4<f32> {
    let ray = reconstruct_world_ray(in.uv, view);
    return vec4<f32>(final_sky_color(ray), 1.0);
}
#else
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let ray = reconstruct_world_ray(in.uv, 0u);
    return vec4<f32>(final_sky_color(ray), 1.0);
}
#endif
