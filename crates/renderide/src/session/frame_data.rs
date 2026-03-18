//! Helpers for processing FrameSubmitData in Session.

use crate::shared::{FrameSubmitData, RenderTransform};

/// Primary view selection from frame data.
#[derive(Debug, Clone)]
pub struct PrimaryViewSelection {
    pub space_id: i32,
    pub override_view_position: bool,
    pub view_position_is_external: bool,
    pub root_transform: RenderTransform,
    pub view_transform: RenderTransform,
}

/// Applies clip planes, FOV, and output state from frame data to the given mutable refs.
pub fn apply_clip_and_output_state(
    data: &FrameSubmitData,
    near_clip: &mut f32,
    far_clip: &mut f32,
    desktop_fov: &mut f32,
    lock_cursor: &mut bool,
) {
    *near_clip = data.near_clip;
    *far_clip = data.far_clip;
    *desktop_fov = data.desktop_fov;
    if let Some(ref output) = data.output_state {
        *lock_cursor = output.lock_cursor;
    }
}

/// Validates that at most one active non-overlay space exists. Returns Err(()) if invalid.
pub fn validate_active_non_overlay(data: &FrameSubmitData) -> Result<(), ()> {
    let active_non_overlay: Vec<_> = data
        .render_spaces
        .iter()
        .filter(|u| u.is_active && !u.is_overlay)
        .collect();
    if active_non_overlay.len() > 1 {
        return Err(());
    }
    Ok(())
}

/// Selects the primary view from frame data.
/// View selection: override (freecam) → overriden_view_transform; else → root_transform.
/// When view_position_is_external is true (e.g. VR/third-person), view may need to come from
/// input/head state; we use root for now since the host does not send a separate head pose.
pub fn select_primary_view(data: &FrameSubmitData) -> Option<PrimaryViewSelection> {
    let active_non_overlay: Vec<_> = data
        .render_spaces
        .iter()
        .filter(|u| u.is_active && !u.is_overlay)
        .collect();

    if let Some(update) = active_non_overlay.first() {
        let view_transform = if update.override_view_position {
            update.overriden_view_transform
        } else {
            update.root_transform
        };
        return Some(PrimaryViewSelection {
            space_id: update.id,
            override_view_position: update.override_view_position,
            view_position_is_external: update.view_position_is_external,
            root_transform: update.root_transform,
            view_transform,
        });
    }

    data.render_spaces.first().map(|first| PrimaryViewSelection {
        space_id: first.id,
        override_view_position: first.override_view_position,
        view_position_is_external: first.view_position_is_external,
        root_transform: first.root_transform,
        view_transform: first.root_transform,
    })
}
