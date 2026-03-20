//! Frustum culling for skinned (bone-deformed) meshes.
//!
//! # Why rigid culling doesn't work for skinned meshes
//!
//! Rigid culling transforms the bind-pose local bounds by the model matrix. For skinned meshes
//! the vertices are deformed by bone matrices — bones may translate, rotate, or scale far from
//! their bind-pose positions, making the bind-pose AABB an unreliable proxy for the on-screen
//! region. Testing only the model-matrix-transformed bind bounds can produce **false negatives**
//! (culling visible skinned draws).
//!
//! # Culling strategy
//!
//! Each bone's **final bone matrix** (produced by
//! [`crate::scene::graph::SceneGraph::compute_bone_matrices`]) is the concatenation of the bone
//! world transform and the inverse bind pose. Applying it to the vector `(0,0,0,1)` gives the
//! world-space position of that bone's bind-space origin — i.e. where a vertex pinned to that
//! bone with zero local offset would land in world space.
//!
//! We therefore:
//!
//! 1. Extract the **translation column** (column 3) of every bone matrix:
//!    `(bone[3][0], bone[3][1], bone[3][2])` — the world position of that bone's origin.
//! 2. Compute the **AABB** of all bone world positions.
//! 3. **Expand** that AABB uniformly by the mesh's largest local half-extent
//!    (`max(|extents.x|, |extents.y|, |extents.z|)`). This conservatively captures mesh surface
//!    geometry that extends beyond joint pivots: a vertex at local bind-space offset `v` from
//!    a bone cannot end up further than `‖v‖` from that bone's world position after skinning.
//! 4. Test the expanded AABB against the view-projection clip volume.
//!
//! # Matrix format
//!
//! Bone matrices are in the column-major layout produced by
//! [`crate::scene::graph::SceneGraph::compute_bone_matrices`] via `glam_mat4_to_bind_pose`:
//! `mat[col][row]`. Column 3 = `mat[3]` = `[tx, ty, tz, 1.0]`.
//!
//! # Conservative behaviour
//!
//! Returns `true` (keep the draw) when:
//! - `bone_matrices` is empty (no data to derive an AABB from).
//! - All bone translations are non-finite (corrupt data).
//!
//! This guarantees no visible skinned draw is incorrectly culled at the cost of some false
//! positives (draw calls that are actually off-screen but pass the test).

use glam::{Mat4, Vec3};

use crate::shared::RenderBoundingBox;

use super::{mesh_bounds_max_half_extent, world_aabb_visible_in_homogeneous_clip};

/// Returns `true` if the skinned mesh draw is potentially visible and should be submitted.
///
/// See the [module documentation](self) for the full culling strategy.
///
/// # Arguments
/// - `bounds`: local-space bounding box from the mesh asset (bind-pose bounds).
///   Used only to compute the expansion radius; degenerate bounds (near-zero extents)
///   result in no expansion, which is still safe (the bone-position AABB is still tested).
/// - `bone_matrices`: final bone matrices in column-major `mat[col][row]` format.
///   Translation column is `mat[3]` = `[tx, ty, tz, 1.0]`.
/// - `view_proj`: the same view-projection matrix used by the mesh pass for this batch.
pub fn skinned_mesh_potentially_visible(
    bounds: &RenderBoundingBox,
    bone_matrices: &[[[f32; 4]; 4]],
    view_proj: Mat4,
) -> bool {
    if bone_matrices.is_empty() {
        // No bones: cannot derive a world AABB; conservatively keep the draw.
        return true;
    }

    // Step 1: build AABB from bone world-space origin positions.
    // Column-major: bone[col][row], so col 3 = [tx, ty, tz, tw].
    let mut world_min = Vec3::splat(f32::INFINITY);
    let mut world_max = Vec3::splat(f32::NEG_INFINITY);

    for bone in bone_matrices {
        let tx = bone[3][0];
        let ty = bone[3][1];
        let tz = bone[3][2];
        if tx.is_finite() && ty.is_finite() && tz.is_finite() {
            let p = Vec3::new(tx, ty, tz);
            world_min = world_min.min(p);
            world_max = world_max.max(p);
        }
    }

    if !world_min.x.is_finite() {
        // All bone positions were non-finite: cannot determine AABB; conservatively keep.
        return true;
    }

    // Step 2: expand AABB by the mesh's largest local half-extent.
    // Skipped when extents are degenerate to avoid inflating with unreliable metadata.
    let expand = mesh_bounds_max_half_extent(bounds);
    if expand > 0.0 {
        world_min -= Vec3::splat(expand);
        world_max += Vec3::splat(expand);
    }

    // Step 3: test the expanded AABB against the frustum.
    world_aabb_visible_in_homogeneous_clip(view_proj, world_min, world_max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::pass::reverse_z_projection;
    use crate::scene::math::matrix_na_to_glam;
    use glam::Mat4;
    use nalgebra::{Matrix4 as NaMat4, Point3, Vector3 as NaVec3};

    fn look_vp_naive() -> Mat4 {
        let view = NaMat4::look_at_rh(
            &Point3::new(0.0, 0.0, 5.0),
            &Point3::new(0.0, 0.0, 0.0),
            &NaVec3::new(0.0, 1.0, 0.0),
        );
        let proj = reverse_z_projection(1.0, 60f32.to_radians(), 0.1, 100.0);
        matrix_na_to_glam(&(proj * view))
    }

    fn make_bounds(ex: f32) -> RenderBoundingBox {
        RenderBoundingBox {
            center: NaVec3::zeros(),
            extents: NaVec3::new(ex, ex, ex),
        }
    }

    /// Column-major bone matrix with translation (tx, ty, tz).
    fn bone_at(tx: f32, ty: f32, tz: f32) -> [[f32; 4]; 4] {
        // col-major: mat[col][row].  col3 = [tx, ty, tz, 1.0].
        [
            [1.0, 0.0, 0.0, 0.0], // col 0
            [0.0, 1.0, 0.0, 0.0], // col 1
            [0.0, 0.0, 1.0, 0.0], // col 2
            [tx, ty, tz, 1.0],    // col 3 = translation
        ]
    }

    // ─── Conservative pass-through ────────────────────────────────────────────

    #[test]
    fn empty_bone_list_conservative() {
        let vp = look_vp_naive();
        assert!(skinned_mesh_potentially_visible(&make_bounds(0.5), &[], vp));
    }

    #[test]
    fn all_nonfinite_bones_conservative() {
        let vp = look_vp_naive();
        let bad = [[f32::NAN; 4]; 4];
        assert!(skinned_mesh_potentially_visible(&make_bounds(0.5), &[bad], vp));
    }

    // ─── Visibility ───────────────────────────────────────────────────────────

    #[test]
    fn bone_at_origin_visible() {
        let vp = look_vp_naive();
        // Camera at (0,0,5) looking at origin. Bone at (0,0,0) with small mesh = visible.
        let bone = bone_at(0.0, 0.0, 0.0);
        assert!(skinned_mesh_potentially_visible(&make_bounds(0.5), &[bone], vp));
    }

    #[test]
    fn bone_behind_camera_culled() {
        let vp = look_vp_naive();
        // Camera at (0,0,5) looking at origin. Bone at z=20 is behind the camera.
        let bone = bone_at(0.0, 0.0, 20.0);
        let b = make_bounds(0.5); // small extents → expand = 0.5, still well behind camera
        assert!(!skinned_mesh_potentially_visible(&b, &[bone], vp));
    }

    #[test]
    fn bone_far_left_culled() {
        let vp = look_vp_naive();
        // Bone 100 units to the left, mesh extents 0.5 → expanded AABB is still off-screen.
        let bone = bone_at(100.0, 0.0, 0.0);
        assert!(!skinned_mesh_potentially_visible(&make_bounds(0.5), &[bone], vp));
    }

    #[test]
    fn multiple_bones_span_frustum() {
        let vp = look_vp_naive();
        // One bone behind camera, one in front: union AABB spans the near plane → keep.
        let b_behind = bone_at(0.0, 0.0, 20.0);
        let b_front = bone_at(0.0, 0.0, 0.0);
        assert!(skinned_mesh_potentially_visible(
            &make_bounds(0.5),
            &[b_behind, b_front],
            vp
        ));
    }

    #[test]
    fn large_extents_rescue_off_center_bone() {
        let vp = look_vp_naive();
        // Bone slightly outside right edge; large mesh extents expand AABB back into frustum.
        // At depth z=0 (4 units from eye) the right frustum clip is ~tan(30°)*4 ≈ 2.3 units.
        // Bone at x=3.5 is outside, but mesh extents=2 expand left by 2 → min_x=1.5, visible.
        let bone = bone_at(3.5, 0.0, 0.0);
        let big_extents = make_bounds(2.0);
        assert!(skinned_mesh_potentially_visible(&big_extents, &[bone], vp));
    }

    #[test]
    fn small_extents_far_bone_still_culled() {
        let vp = look_vp_naive();
        // Bone 100 units right, tiny mesh → cannot rescue.
        let bone = bone_at(100.0, 0.0, 0.0);
        assert!(!skinned_mesh_potentially_visible(&make_bounds(0.01), &[bone], vp));
    }
}
