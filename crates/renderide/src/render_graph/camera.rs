//! Host-facing view and **reverse-Z** projection (`crates_old` `reverse_z_projection` / `ViewParams`).
//!
//! Matches the legacy mesh pass: world-to-view applies a Z flip for Vulkan/WebGPU clip, and
//! perspective uses vertical FOV in **radians** with clip planes from [`super::frame_params::HostCameraFrame`].

use glam::{Mat4, Vec3, Vec4};

use crate::scene::render_transform_to_matrix;
use crate::shared::RenderTransform;

/// Clamps scale for view matrix construction: if any axis is nearly zero, use unit scale (legacy `filter_scale`).
fn filter_scale_for_view(scale: Vec3) -> Vec3 {
    const MIN: f32 = 1e-8;
    if scale.x.abs() < MIN || scale.y.abs() < MIN || scale.z.abs() < MIN {
        Vec3::splat(1.0)
    } else {
        scale
    }
}

/// Z-flip for RH engine space → Vulkan/WebGPU-style clip (legacy `apply_view_handedness_fix`).
#[inline]
fn apply_view_handedness_fix(view: Mat4) -> Mat4 {
    let z_flip = Mat4::from_scale(Vec3::new(1.0, 1.0, -1.0));
    z_flip * view
}

/// World-to-view matrix from a host [`RenderTransform`] (camera / eye TRS).
///
/// Applies legacy scale filtering and handedness fix so `view_proj * world_pos` matches the old mesh path.
pub fn view_matrix_from_render_transform(tr: &RenderTransform) -> Mat4 {
    let mut t = *tr;
    let fs = filter_scale_for_view(Vec3::new(tr.scale.x, tr.scale.y, tr.scale.z));
    t.scale.x = fs.x;
    t.scale.y = fs.y;
    t.scale.z = fs.z;
    let cam = render_transform_to_matrix(&t);
    apply_view_handedness_fix(cam.inverse())
}

/// Reverse-Z perspective projection (column-major [`Mat4`], same coefficients as legacy nalgebra path).
///
/// * `vertical_fov` — vertical field of view in **radians**
/// * `near` / `far` — positive distances (`far > near`)
pub fn reverse_z_perspective(aspect: f32, vertical_fov: f32, near: f32, far: f32) -> Mat4 {
    let vertical_half = vertical_fov / 2.0;
    let tan_vertical_half = vertical_half.tan();
    let horizontal_fov = (tan_vertical_half * aspect)
        .atan()
        .clamp(0.1_f32, std::f32::consts::FRAC_PI_2 - 0.1)
        * 2.0;
    let tan_horizontal_half = (horizontal_fov / 2.0).tan();
    let f_x = 1.0 / tan_horizontal_half;
    let f_y = 1.0 / tan_vertical_half;
    let z2 = near / (far - near);
    let z3 = (far * near) / (far - near);
    Mat4::from_cols(
        Vec4::new(f_x, 0.0, 0.0, 0.0),
        Vec4::new(0.0, f_y, 0.0, 0.0),
        Vec4::new(0.0, 0.0, z2, -1.0),
        Vec4::new(0.0, 0.0, z3, 0.0),
    )
}

/// Reverse-Z orthographic projection (`half_width`, `half_height` in view space).
pub fn reverse_z_orthographic(half_width: f32, half_height: f32, near: f32, far: f32) -> Mat4 {
    let range = far - near;
    let z_scale = -2.0 / range;
    let z_offset = (far + near) / range;
    Mat4::from_cols(
        Vec4::new(1.0 / half_width, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0 / half_height, 0.0, 0.0),
        Vec4::new(0.0, 0.0, z_scale, 0.0),
        Vec4::new(0.0, 0.0, z_offset, 1.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reverse_z_perspective_finite_diagonal() {
        let m = reverse_z_perspective(16.0 / 9.0, 60f32.to_radians(), 0.1, 500.0);
        assert!(m.w_axis.w.is_finite());
        assert!(m.x_axis.x > 0.0 && m.y_axis.y > 0.0);
        assert!(m.z_axis.w == -1.0);
    }

    #[test]
    fn view_handedness_applies_z_flip() {
        let tr = RenderTransform::default();
        let v = view_matrix_from_render_transform(&tr);
        let z_flip = Mat4::from_scale(Vec3::new(1.0, 1.0, -1.0));
        let unflipped = render_transform_to_matrix(&tr).inverse();
        let expected = z_flip * unflipped;
        assert!(
            (v - expected)
                .to_cols_array()
                .iter()
                .map(|x| x.abs())
                .sum::<f32>()
                < 1e-5
        );
    }

    #[test]
    fn orthographic_matches_legacy_z_coeffs() {
        let m = reverse_z_orthographic(2.0, 1.0, 0.05, 100.0);
        let range = 100.0 - 0.05;
        let z_scale = -2.0 / range;
        let z_off = (100.0 + 0.05) / range;
        assert!((m.z_axis.z - z_scale).abs() < 1e-5);
        assert!((m.w_axis.z - z_off).abs() < 1e-5);
    }
}
