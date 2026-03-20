//! Render configuration types and application-level settings.
//!
//! ## Config loading precedence
//!
//! 1. **Defaults** — hardcoded values in [`Default`] impls.
//! 2. **`configuration.ini`** — searched next to the executable, then in the
//!    current working directory (see [`find_config_ini`]).
//! 3. **Env vars** — highest priority, override everything
//!    (e.g. `RENDERIDE_NO_RTAO=1`).
//!
//! Two separate structs are exposed:
//! - [`AppConfig`]  — client-side settings (FPS caps, HUD toggle).  Read once
//!   at startup in `app.rs`; never touched by IPC session commands.
//! - [`RenderConfig`] — rendering parameters that *can* be overridden by host
//!   IPC commands (vsync, RTAO, clip planes, …).  Loaded via [`RenderConfig::load`].

use std::path::PathBuf;

// ─── INI parser ───────────────────────────────────────────────────────────────

/// Searches for `configuration.ini` next to the running executable, then in
/// the current working directory.  Returns the first path that exists.
pub fn find_config_ini() -> Option<PathBuf> {
    // 1. Directory containing the running executable (good for release builds).
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join("configuration.ini");
            if p.exists() {
                return Some(p);
            }
        }
    }
    // 2. Current working directory (good for `cargo run` from the repo root).
    if let Ok(cwd) = std::env::current_dir() {
        let p = cwd.join("configuration.ini");
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Parses `content` as a simple INI file.
///
/// Returns `(section, key, value)` triples where both `section` and `key` are
/// already lower-cased.  Lines beginning with `#` or `;` are comments.
/// Inline comments (after `#` or `;`) are stripped from values.
fn parse_ini(content: &str) -> Vec<(String, String, String)> {
    let mut result = Vec::new();
    let mut section = String::new();
    for raw in content.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        if line.starts_with('[') {
            if let Some(end) = line.find(']') {
                section = line[1..end].trim().to_lowercase();
            }
            continue;
        }
        if let Some(eq) = line.find('=') {
            let key = line[..eq].trim().to_lowercase();
            let raw_val = line[eq + 1..].trim();
            // Strip inline comments after `#` or `;`.
            let val = raw_val
                .split_once('#')
                .map(|(v, _)| v)
                .or_else(|| raw_val.split_once(';').map(|(v, _)| v))
                .unwrap_or(raw_val)
                .trim();
            result.push((section.clone(), key, val.to_string()));
        }
    }
    result
}

/// Parses boolean-like strings: `true/false`, `1/0`, `yes/no`, `on/off`.
fn parse_bool(s: &str) -> Option<bool> {
    match s.trim().to_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Some(true),
        "false" | "0" | "no" | "off" => Some(false),
        _ => None,
    }
}

// ─── AppConfig ────────────────────────────────────────────────────────────────

/// Client-side application settings loaded from `configuration.ini`.
///
/// These are *not* sent over IPC and will never be overridden by host commands.
/// Use them to control frame-rate limits and the debug HUD.
#[derive(Clone, Debug)]
pub struct AppConfig {
    /// Maximum frames per second while the window is **focused** (`0` = uncapped).
    pub focused_fps: u32,
    /// Maximum frames per second while the window is **unfocused** / tabbed out
    /// (`0` = uncapped).
    pub unfocused_fps: u32,
    /// Show the in-process debug HUD overlay.  Set to `false` to hide it and
    /// avoid any associated GPU overhead.
    pub show_hud: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            focused_fps: 240,
            unfocused_fps: 60,
            show_hud: true,
        }
    }
}

impl AppConfig {
    /// Loads [`AppConfig`] from `configuration.ini` if found, otherwise returns
    /// [`Default::default`].
    pub fn load() -> Self {
        let mut cfg = Self::default();
        let Some(path) = find_config_ini() else {
            return cfg;
        };
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!(
                    "[renderide] configuration.ini read error ({}): {}",
                    path.display(),
                    e
                );
                return cfg;
            }
        };
        for (section, key, value) in parse_ini(&content) {
            match (section.as_str(), key.as_str()) {
                ("display", "focused_fps") => {
                    if let Ok(v) = value.parse::<u32>() {
                        cfg.focused_fps = v;
                    }
                }
                ("display", "unfocused_fps") => {
                    if let Ok(v) = value.parse::<u32>() {
                        cfg.unfocused_fps = v;
                    }
                }
                ("hud", "show_hud") => {
                    if let Some(v) = parse_bool(&value) {
                        cfg.show_hud = v;
                    }
                }
                _ => {}
            }
        }
        cfg
    }
}

// ─── RenderConfig ─────────────────────────────────────────────────────────────

/// Render configuration (clip planes, FOV, display settings).
#[derive(Clone, Debug)]
pub struct RenderConfig {
    /// Near clip plane distance.
    pub near_clip: f32,
    /// Far clip plane distance.
    pub far_clip: f32,
    /// Desktop field of view in degrees.
    pub desktop_fov: f32,
    /// Whether vertical sync is enabled.
    pub vsync: bool,
    /// When true, use UV debug pipeline for meshes that have UVs.
    pub use_debug_uv: bool,
    /// When true, main scene meshes use PBR pipeline instead of NormalDebug. Default true.
    pub use_pbr: bool,
    /// When true, apply the mesh root (drawable's model_matrix) to skinned MVP.
    /// Matches Unity SkinnedMeshRenderer: vertices are in mesh root local space.
    pub skinned_apply_mesh_root_transform: bool,
    /// When true, use root_bone_transform_id from BoneAssignment for root-relative bone matrices.
    /// Enables A/B testing of coordinate alignment. Default false.
    pub skinned_use_root_bone: bool,
    /// When true, log diagnostic info for the first skinned draw each frame.
    pub debug_skinned: bool,
    /// When true, log blendshape batch count and first few weights each frame.
    /// Can be enabled via RENDERIDE_DEBUG_BLENDSHAPES=1.
    pub debug_blendshapes: bool,
    /// When true, apply an extra Z flip to skinned MVP for handedness correction.
    /// Use when skinned meshes appear mirrored vs non-skinned. Default false.
    pub skinned_flip_handedness: bool,
    /// When true and ray tracing is available, RTAO (Ray-Traced Ambient Occlusion) may be used.
    /// Toggle for A/B testing. Default true.
    pub rtao_enabled: bool,
    /// RTAO strength: how much occlusion darkens the scene. 0 = no effect, 1 = full darkening.
    /// Default 0.5.
    pub rtao_strength: f32,
    /// RTAO ray max distance in world units. Rays beyond this are not considered occluded.
    /// Default 1.0.
    pub ao_radius: f32,
    /// When true, mesh draws outside the view frustum are skipped on the CPU: rigid meshes use
    /// local bounds transformed by the model matrix; non-overlay skinned meshes use a conservative
    /// world AABB derived from bone world origins (see [`crate::render::visibility::skinned`]).
    /// Default true.
    pub frustum_culling: bool,
    /// Reserved for future per-batch mesh-draw worker threads. Not active while [`crate::session::Session`]
    /// is not [`Sync`] (IPC). Disable with `RENDERIDE_PARALLEL_MESH_PREP=0` to match future defaults.
    pub parallel_mesh_draw_prep_batches: bool,
}

impl RenderConfig {
    /// Loads config from defaults → `configuration.ini` → env vars.
    ///
    /// **INI keys** (under their respective sections):
    /// - `[display]` `vsync`
    /// - `[rendering]` `rtao_enabled`, `rtao_strength`, `ao_radius`, `frustum_culling`
    ///
    /// **Env vars** (highest priority):
    /// - `RENDERIDE_DEBUG_BLENDSHAPES=1` — enables blendshape debug logging.
    /// - `RENDERIDE_NO_FRUSTUM_CULL=1`  — disables CPU frustum culling.
    /// - `RENDERIDE_PARALLEL_MESH_PREP=0` — disables parallel mesh-draw prep.
    /// - `RENDERIDE_NO_RTAO=1`           — disables RTAO even when RT is available.
    pub fn load() -> Self {
        let mut config = Self::default();

        // Layer 2: configuration.ini overrides.
        if let Some(path) = find_config_ini() {
            if let Ok(content) = std::fs::read_to_string(&path) {
                for (section, key, value) in parse_ini(&content) {
                    match (section.as_str(), key.as_str()) {
                        ("display", "vsync") => {
                            if let Some(v) = parse_bool(&value) {
                                config.vsync = v;
                            }
                        }
                        ("rendering", "rtao_enabled") => {
                            if let Some(v) = parse_bool(&value) {
                                config.rtao_enabled = v;
                            }
                        }
                        ("rendering", "rtao_strength") => {
                            if let Ok(v) = value.parse::<f32>() {
                                config.rtao_strength = v;
                            }
                        }
                        ("rendering", "ao_radius") => {
                            if let Ok(v) = value.parse::<f32>() {
                                config.ao_radius = v;
                            }
                        }
                        ("rendering", "frustum_culling") => {
                            if let Some(v) = parse_bool(&value) {
                                config.frustum_culling = v;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Layer 3: env var overrides (highest priority).
        if std::env::var("RENDERIDE_DEBUG_BLENDSHAPES").as_deref() == Ok("1") {
            config.debug_blendshapes = true;
        }
        if std::env::var("RENDERIDE_NO_FRUSTUM_CULL").as_deref() == Ok("1") {
            config.frustum_culling = false;
        }
        if std::env::var("RENDERIDE_PARALLEL_MESH_PREP").as_deref() == Ok("0") {
            config.parallel_mesh_draw_prep_batches = false;
        }
        if std::env::var("RENDERIDE_NO_RTAO").as_deref() == Ok("1") {
            config.rtao_enabled = false;
        }
        config
    }
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            near_clip: 0.01,
            far_clip: 1024.0,
            desktop_fov: 75.0,
            vsync: false,
            use_debug_uv: false,
            use_pbr: true,
            skinned_apply_mesh_root_transform: true,
            skinned_use_root_bone: false,
            debug_skinned: false,
            debug_blendshapes: false,
            skinned_flip_handedness: false,
            rtao_enabled: true,
            rtao_strength: 1.0,
            ao_radius: 1.0,
            frustum_culling: true,
            parallel_mesh_draw_prep_batches: true,
        }
    }
}
