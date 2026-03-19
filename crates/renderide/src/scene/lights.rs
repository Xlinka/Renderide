//! Light cache and resolved light types for rendering.
//!
//! Stores light data per space, merges with scene transforms, and produces
//! world-space resolved lights for the render loop.

use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};

use crate::shared::{LightData, LightType, LightsBufferRendererState};

/// Cached light entry combining pose data from submission with state from updates.
#[derive(Clone, Debug)]
pub struct CachedLight {
    /// Local-space pose and color from LightsBufferRendererSubmission.
    pub data: LightData,
    /// Renderable index, type, and shadow params from LightsBufferRendererUpdate.
    pub state: LightsBufferRendererState,
}

impl CachedLight {
    /// Creates a new cached light with default state when only data is available.
    pub fn from_data(data: LightData) -> Self {
        Self {
            data,
            state: LightsBufferRendererState::default(),
        }
    }
}

/// Resolved light in world space, ready for the render loop.
#[derive(Clone, Debug)]
pub struct ResolvedLight {
    /// World-space position.
    pub world_position: Vec3,
    /// World-space direction (normalized; -Z for spot/directional).
    pub world_direction: Vec3,
    /// RGB color.
    pub color: Vec3,
    /// Light intensity.
    pub intensity: f32,
    /// Attenuation range (point/spot).
    pub range: f32,
    /// Spot angle in degrees (spot only).
    pub spot_angle: f32,
    /// Light type: point, directional, or spot.
    pub light_type: LightType,
    /// Global unique ID for consumed feedback.
    pub global_unique_id: i32,
}

/// Cache of lights per lights buffer (keyed by lights_buffer_unique_id).
/// Uses 1:1 mapping: lights_buffer_unique_id == space_id.
#[derive(Clone, Debug)]
pub struct LightCache {
    /// Full light data per buffer ID. Populated from LightsBufferRendererSubmission.
    buffers: HashMap<i32, Vec<LightData>>,
    /// Per-space light entries (data + state). Index in vec matches slot.
    /// Key: space_id. Populated from LightsBufferRendererUpdate.
    spaces: HashMap<i32, Vec<CachedLight>>,
}

impl LightCache {
    /// Creates a new empty light cache.
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            spaces: HashMap::new(),
        }
    }

    /// Stores full light data from a submission. Overwrites any existing buffer.
    pub fn store_full(&mut self, lights_buffer_unique_id: i32, light_data: Vec<LightData>) {
        self.buffers.insert(lights_buffer_unique_id, light_data);
    }

    /// Applies incremental update (states, removals, additions) for a space.
    /// Uses lights_buffer_unique_id == space_id to look up light data from store_full.
    pub fn apply_update(
        &mut self,
        space_id: i32,
        removals: &[i32],
        additions: &[LightsBufferRendererState],
        states: &[LightsBufferRendererState],
    ) {
        let light_data = self.buffers.get(&space_id).map(|v| v.as_slice()).unwrap_or(&[]);
        let entries = self.spaces.entry(space_id).or_default();

        // Apply removals (descending order for swap_remove)
        let mut indices: Vec<usize> = removals
            .iter()
            .take_while(|&&i| i >= 0)
            .map(|&i| i as usize)
            .collect();
        indices.sort_by(|a, b| b.cmp(a));
        for idx in indices {
            if idx < entries.len() {
                entries.swap_remove(idx);
            }
        }

        // Apply additions: new slots get LightData from submission by index
        for (i, state) in additions
            .iter()
            .enumerate()
            .take_while(|(_, s)| s.renderable_index >= 0)
        {
            let idx = entries.len() + i;
            let data = light_data.get(idx).cloned().unwrap_or_default();
            entries.push(CachedLight {
                data,
                state: *state,
            });
        }

        // Apply state updates
        for state in states {
            if state.renderable_index < 0 {
                break;
            }
            let idx = state.renderable_index as usize;
            if idx < entries.len() {
                entries[idx].state = *state;
                if let Some(d) = light_data.get(idx) {
                    entries[idx].data = *d;
                }
            }
        }
    }

    /// Returns cached lights for a space. Call after apply_update.
    pub fn get_lights_for_space(&self, space_id: i32) -> Option<&[CachedLight]> {
        self.spaces.get(&space_id).map(|v| v.as_slice())
    }

    /// Returns mutable cached lights for a space.
    pub fn get_lights_for_space_mut(&mut self, space_id: i32) -> Option<&mut Vec<CachedLight>> {
        self.spaces.get_mut(&space_id)
    }

    /// Removes a space's lights (e.g. when space is removed).
    pub fn remove_space(&mut self, space_id: i32) {
        self.spaces.remove(&space_id);
    }

    /// Resolves cached lights to world space using scene world matrices.
    pub fn resolve_lights(
        &self,
        space_id: i32,
        get_world_matrix: impl Fn(usize) -> Option<Mat4>,
    ) -> Vec<ResolvedLight> {
        let Some(lights) = self.get_lights_for_space(space_id) else {
            return Vec::new();
        };

        const FORWARD: Vec3 = Vec3::new(0.0, 0.0, -1.0);

        let mut resolved = Vec::with_capacity(lights.len());
        for cached in lights {
            let world = if cached.state.renderable_index >= 0 {
                get_world_matrix(cached.state.renderable_index as usize)
                    .unwrap_or(Mat4::IDENTITY)
            } else {
                Mat4::IDENTITY
            };

            let point = cached.data.point;
            let p = Vec3::new(point.x, point.y, point.z);
            let world_pos = world.transform_point3(p);

            let ori = cached.data.orientation;
            let q = Quat::from_xyzw(ori.i, ori.j, ori.k, ori.w);
            let world_dir = (world.to_scale_rotation_translation().1 * q) * FORWARD;
            let world_dir = if world_dir.length_squared() > 1e-10 {
                world_dir.normalize()
            } else {
                Vec3::NEG_Z
            };

            let color = cached.data.color;
            let color = Vec3::new(color.x, color.y, color.z);

            resolved.push(ResolvedLight {
                world_position: world_pos,
                world_direction: world_dir,
                color,
                intensity: cached.data.intensity,
                range: cached.data.range,
                spot_angle: cached.data.angle,
                light_type: cached.state.light_type,
                global_unique_id: cached.state.global_unique_id,
            });
        }
        resolved
    }
}

impl Default for LightCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::ShadowType;
    use nalgebra::{Quaternion, Vector3};

    fn make_light_data(pos: (f32, f32, f32), color: (f32, f32, f32)) -> LightData {
        LightData {
            point: Vector3::new(pos.0, pos.1, pos.2),
            orientation: Quaternion::identity(),
            color: Vector3::new(color.0, color.1, color.2),
            intensity: 1.0,
            range: 10.0,
            angle: 45.0,
        }
    }

    fn make_state(renderable_index: i32, global_unique_id: i32, light_type: LightType) -> LightsBufferRendererState {
        LightsBufferRendererState {
            renderable_index,
            global_unique_id,
            shadow_strength: 0.0,
            shadow_near_plane: 0.0,
            shadow_map_resolution: 0,
            shadow_bias: 0.0,
            shadow_normal_bias: 0.0,
            cookie_texture_asset_id: -1,
            light_type,
            shadow_type: ShadowType::none,
            _padding: [0; 2],
        }
    }

    #[test]
    fn test_light_cache_store_full_and_apply_additions() {
        let mut cache = LightCache::new();
        let space_id = 0;
        let light_data = vec![
            make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            make_light_data((0.0, 2.0, 0.0), (0.0, 1.0, 0.0)),
        ];
        cache.store_full(space_id, light_data);

        let additions = vec![
            make_state(0, 100, LightType::point),
            make_state(1, 101, LightType::directional),
        ];
        cache.apply_update(space_id, &[], &additions, &[]);

        let lights = cache.get_lights_for_space(space_id).unwrap();
        assert_eq!(lights.len(), 2);
        assert_eq!(lights[0].data.point.x, 1.0);
        assert_eq!(lights[0].state.global_unique_id, 100);
        assert_eq!(lights[1].state.light_type, LightType::directional);
    }

    #[test]
    fn test_light_cache_removals() {
        let mut cache = LightCache::new();
        let space_id = 0;
        let light_data = vec![
            make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            make_light_data((0.0, 2.0, 0.0), (0.0, 1.0, 0.0)),
            make_light_data((0.0, 0.0, 3.0), (0.0, 0.0, 1.0)),
        ];
        cache.store_full(space_id, light_data);

        let additions = vec![
            make_state(0, 100, LightType::point),
            make_state(1, 101, LightType::point),
            make_state(2, 102, LightType::point),
        ];
        cache.apply_update(space_id, &[], &additions, &[]);
        assert_eq!(cache.get_lights_for_space(space_id).unwrap().len(), 3);

        cache.apply_update(space_id, &[1], &[], &[]);
        let lights = cache.get_lights_for_space(space_id).unwrap();
        assert_eq!(lights.len(), 2);
        assert_eq!(lights[0].state.global_unique_id, 100);
        assert_eq!(lights[1].state.global_unique_id, 102);
    }

    #[test]
    fn test_light_cache_resolve_world_space() {
        let mut cache = LightCache::new();
        let space_id = 0;
        let light_data = vec![make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))];
        cache.store_full(space_id, light_data);

        let additions = vec![make_state(0, 100, LightType::point)];
        cache.apply_update(space_id, &[], &additions, &[]);

        let world_matrix = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0));
        let resolved = cache.resolve_lights(space_id, |tid| {
            if tid == 0 {
                Some(world_matrix)
            } else {
                None
            }
        });

        assert_eq!(resolved.len(), 1);
        assert!((resolved[0].world_position.x - 11.0).abs() < 1e-5);
        assert!((resolved[0].world_position.y - 0.0).abs() < 1e-5);
        assert!((resolved[0].world_position.z - 0.0).abs() < 1e-5);
    }
}
