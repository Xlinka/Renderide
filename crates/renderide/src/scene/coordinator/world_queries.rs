//! World-matrix queries with render-context overrides and overlay re-rooting.

use glam::{Mat4, Vec3};

use crate::shared::RenderingContext;

use super::super::ids::RenderSpaceId;
use super::super::math::render_transform_has_degenerate_scale;
use super::super::render_space::RenderSpaceState;
use super::super::render_transform_to_matrix;
use super::SceneCoordinator;

impl SceneCoordinator {
    /// Hierarchy world matrix with active render-context-local transform overrides applied.
    pub fn world_matrix_for_context(
        &self,
        id: RenderSpaceId,
        transform_index: usize,
        context: RenderingContext,
    ) -> Option<Mat4> {
        let space = self.spaces.get(&id)?;
        if transform_index >= space.nodes.len() {
            return None;
        }
        if !space.has_transform_overrides_in_context(context) {
            return self.world_matrix(id, transform_index);
        }

        let mut path = Vec::with_capacity(64);
        let mut cursor = transform_index;
        let mut broke = false;
        let mut any_override = false;
        for _ in 0..space.nodes.len() {
            path.push(cursor);
            any_override |= space
                .overridden_local_transform(cursor as i32, context)
                .is_some();
            let parent = *space.node_parents.get(cursor).unwrap_or(&-1);
            if parent < 0 || parent as usize >= space.nodes.len() || parent == cursor as i32 {
                broke = true;
                break;
            }
            cursor = parent as usize;
        }
        if !broke || !any_override {
            return self.world_matrix(id, transform_index);
        }

        let mut world = Mat4::IDENTITY;
        while let Some(node_id) = path.pop() {
            let local = space
                .overridden_local_transform(node_id as i32, context)
                .unwrap_or(space.nodes[node_id]);
            world *= render_transform_to_matrix(&local);
        }
        Some(world)
    }

    /// Hierarchy world matrix prepared for actual rendering.
    ///
    /// Overlay spaces are re-rooted against the current `HeadOutput.transform` before drawing
    /// (`RenderSpace.UpdateOverlayPositioning` on the host side).
    pub fn world_matrix_for_render_context(
        &self,
        id: RenderSpaceId,
        transform_index: usize,
        context: RenderingContext,
        head_output_transform: Mat4,
    ) -> Option<Mat4> {
        let local = self.world_matrix_for_context(id, transform_index, context)?;
        let space = self.spaces.get(&id)?;
        if !space.is_overlay {
            return Some(local);
        }
        Some(overlay_space_root_matrix(space, head_output_transform) * local)
    }

    /// Returns whether the cached hierarchy for `transform_index` contains degenerate object scale.
    ///
    /// Missing spaces or transforms return `false` so callers preserve existing draw fallbacks when
    /// the world cache is unavailable for reasons unrelated to scale.
    pub fn transform_has_degenerate_scale(
        &self,
        id: RenderSpaceId,
        transform_index: usize,
    ) -> bool {
        self.world_caches
            .get(&id)
            .and_then(|cache| cache.degenerate_scales.get(transform_index))
            .copied()
            .unwrap_or(false)
    }

    /// Returns whether the effective render-context transform chain collapses object scale.
    ///
    /// Render-transform overrides replace local TRS components for the selected context. Overlay
    /// head-output re-rooting is intentionally ignored because the visibility decision belongs to
    /// the object transform chain, not the camera/root transform used for view placement.
    pub fn transform_has_degenerate_scale_for_context(
        &self,
        id: RenderSpaceId,
        transform_index: usize,
        context: RenderingContext,
    ) -> bool {
        let Some(space) = self.spaces.get(&id) else {
            return false;
        };
        if transform_index >= space.nodes.len() {
            return false;
        }
        if !space.has_transform_overrides_in_context(context) {
            return self.transform_has_degenerate_scale(id, transform_index);
        }

        let mut path = Vec::with_capacity(64);
        let mut cursor = transform_index;
        let mut broke = false;
        let mut any_override = false;
        for _ in 0..space.nodes.len() {
            path.push(cursor);
            any_override |= space
                .overridden_local_transform(cursor as i32, context)
                .is_some();
            let parent = *space.node_parents.get(cursor).unwrap_or(&-1);
            if parent < 0 || parent as usize >= space.nodes.len() || parent == cursor as i32 {
                broke = true;
                break;
            }
            cursor = parent as usize;
        }
        if !broke || !any_override {
            return self.transform_has_degenerate_scale(id, transform_index);
        }

        while let Some(node_id) = path.pop() {
            let local = space
                .overridden_local_transform(node_id as i32, context)
                .unwrap_or(space.nodes[node_id]);
            if render_transform_has_degenerate_scale(&local) {
                return true;
            }
        }
        false
    }
}

fn overlay_space_root_matrix(space: &RenderSpaceState, head_output_transform: Mat4) -> Mat4 {
    let (scale, rotation, position) = head_output_transform.to_scale_rotation_translation();
    let scale = filter_overlay_scale(scale);
    let position = position - space.root_transform.position;
    let rotation = rotation * space.root_transform.rotation;
    Mat4::from_scale_rotation_translation(scale, rotation, position)
}

fn filter_overlay_scale(scale: Vec3) -> Vec3 {
    if scale.x.min(scale.y).min(scale.z) <= 1e-8 {
        Vec3::ONE
    } else {
        scale
    }
}
