// WGSL equivalent seed for third_party/Resonite.UnityShaders/Assets/Shaders/Common/PBSMetallic.shader
// Property-name parity target:
// _Color _MainTex _Cutoff _Glossiness _GlossMapScale _SmoothnessTextureChannel _Metallic _MetallicGlossMap
// _SpecularHighlights _GlossyReflections _BumpScale _BumpMap _Parallax _ParallaxMap _OcclusionStrength _OcclusionMap
// _EmissionColor _EmissionMap _DetailMask _DetailAlbedoMap _DetailNormalMapScale _DetailNormalMap _UVSec

#import renderide_uniform_ring

struct PbsMetallicMaterialUniform {
    _Color: vec4f,
    _EmissionColor: vec4f,
    _MainTex_ST: vec4f,
    _MetallicGlossMap_ST: vec4f,
    _BumpMap_ST: vec4f,
    _OcclusionMap_ST: vec4f,
    _EmissionMap_ST: vec4f,
    _DetailAlbedoMap_ST: vec4f,
    _DetailNormalMap_ST: vec4f,
    _Cutoff: f32,
    _Glossiness: f32,
    _GlossMapScale: f32,
    _SmoothnessTextureChannel: f32,
    _Metallic: f32,
    _BumpScale: f32,
    _Parallax: f32,
    _OcclusionStrength: f32,
    _DetailNormalMapScale: f32,
    _UVSec: f32,
    _Flags: u32,
    _Pad0: vec3u,
}

const FLAG_NORMALMAP: u32 = 1u;
const FLAG_ALPHATEST_ON: u32 = 2u;
const FLAG_EMISSION: u32 = 4u;
const FLAG_METALLICGLOSSMAP: u32 = 8u;
const FLAG_SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A: u32 = 16u;
const FLAG_SPECULARHIGHLIGHTS_OFF: u32 = 32u;
const FLAG_GLOSSYREFLECTIONS_OFF: u32 = 64u;
const FLAG_PARALLAXMAP: u32 = 128u;
const FLAG_DETAIL_MULX2: u32 = 256u;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv0: vec2f,
    @location(3) tangent: vec4f,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) world_position: vec3f,
    @location(1) world_normal: vec3f,
    @location(2) uv0: vec2f,
}

@group(0) @binding(0) var<uniform> uniforms: array<renderide_uniform_ring::UniformsSlot, 64>;
@group(1) @binding(0) var<uniform> material: PbsMetallicMaterialUniform;
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
@group(1) @binding(2) var _MainTex_sampler: sampler;
@group(1) @binding(3) var _MetallicGlossMap: texture_2d<f32>;
@group(1) @binding(4) var _MetallicGlossMap_sampler: sampler;
@group(1) @binding(5) var _BumpMap: texture_2d<f32>;
@group(1) @binding(6) var _BumpMap_sampler: sampler;
@group(1) @binding(7) var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(8) var _OcclusionMap_sampler: sampler;
@group(1) @binding(9) var _EmissionMap: texture_2d<f32>;
@group(1) @binding(10) var _EmissionMap_sampler: sampler;

fn pow5(x: f32) -> f32 {
    let x2 = x * x;
    return x2 * x2 * x;
}

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(denom * denom * 3.14159265, 0.0001);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = r * r / 8.0;
    return n_dot_v / max(n_dot_v * (1.0 - k) + k, 0.0001);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3f) -> vec3f {
    return f0 + (1.0 - f0) * pow5(1.0 - cos_theta);
}

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let u = uniforms[instance_index];
    let world_pos = u.model * vec4f(in.position, 1.0);
    var out: VertexOutput;
    out.clip_position = u.mvp * vec4f(in.position, 1.0);
    out.world_position = world_pos.xyz;
    out.world_normal = normalize((u.model * vec4f(in.normal, 0.0)).xyz);
    out.uv0 = in.uv0;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let uv = in.uv0 * material._MainTex_ST.xy + material._MainTex_ST.zw;
    let albedo_sample = textureSample(_MainTex, _MainTex_sampler, uv);
    var alpha = albedo_sample.a * material._Color.a;
    if ((material._Flags & FLAG_ALPHATEST_ON) != 0u && alpha - material._Cutoff <= 0.0) {
        discard;
    }

    var base_color = albedo_sample.rgb * material._Color.rgb;
    var metallic = material._Metallic;
    var smoothness = material._Glossiness;

    if ((material._Flags & FLAG_METALLICGLOSSMAP) != 0u) {
        let metal_uv = in.uv0 * material._MetallicGlossMap_ST.xy + material._MetallicGlossMap_ST.zw;
        let mg = textureSample(_MetallicGlossMap, _MetallicGlossMap_sampler, metal_uv);
        metallic *= mg.r;
        if ((material._Flags & FLAG_SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A) != 0u) {
            smoothness = material._GlossMapScale * albedo_sample.a;
        } else {
            smoothness = material._GlossMapScale * mg.a;
        }
    }

    let roughness = clamp(1.0 - smoothness, 0.045, 1.0);
    let n = normalize(in.world_normal);
    let v = normalize(-in.world_position);
    let l = normalize(vec3f(0.4, 0.8, 0.3));
    let h = normalize(v + l);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 0.0001);
    let n_dot_h = max(dot(n, h), 0.0);
    let f0 = mix(vec3f(0.04, 0.04, 0.04), base_color, metallic);
    let f = fresnel_schlick(max(dot(h, v), 0.0), f0);
    let d = distribution_ggx(n_dot_h, roughness);
    let g = geometry_smith(n_dot_v, n_dot_l, roughness);
    let specular = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 0.0001);
    let kd = (1.0 - f) * (1.0 - metallic);
    var lighting = (kd * base_color / 3.14159265 + specular) * n_dot_l;

    let occ_uv = in.uv0 * material._OcclusionMap_ST.xy + material._OcclusionMap_ST.zw;
    let occ = textureSample(_OcclusionMap, _OcclusionMap_sampler, occ_uv).g;
    let occlusion = mix(1.0, occ, material._OcclusionStrength);
    lighting *= occlusion;

    if ((material._Flags & FLAG_EMISSION) != 0u) {
        let emission_uv = in.uv0 * material._EmissionMap_ST.xy + material._EmissionMap_ST.zw;
        let emission = textureSample(_EmissionMap, _EmissionMap_sampler, emission_uv).rgb * material._EmissionColor.rgb;
        lighting += emission;
    }

    return vec4f(lighting, alpha);
}
