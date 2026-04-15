//! Path resolution for [`super::openxr_actions::create_openxr_input_parts`].

use openxr as xr;

/// User hand paths and interaction profile [`xr::Path`] handles.
pub(super) struct UserAndProfilePaths {
    pub left_user_path: xr::Path,
    pub right_user_path: xr::Path,
    pub oculus_touch_profile: xr::Path,
    pub valve_index_profile: xr::Path,
    pub htc_vive_profile: xr::Path,
    pub microsoft_motion_profile: xr::Path,
    pub generic_controller_profile: xr::Path,
    pub simple_controller_profile: xr::Path,
    pub pico4_controller_profile: xr::Path,
    pub pico_neo3_controller_profile: xr::Path,
}

/// Resolves `/user/hand/*` and `/interaction_profiles/*` paths.
pub(super) fn resolve_user_and_profile_paths(
    instance: &xr::Instance,
) -> Result<UserAndProfilePaths, xr::sys::Result> {
    Ok(UserAndProfilePaths {
        left_user_path: instance.string_to_path("/user/hand/left")?,
        right_user_path: instance.string_to_path("/user/hand/right")?,
        oculus_touch_profile: instance
            .string_to_path("/interaction_profiles/oculus/touch_controller")?,
        valve_index_profile: instance
            .string_to_path("/interaction_profiles/valve/index_controller")?,
        htc_vive_profile: instance.string_to_path("/interaction_profiles/htc/vive_controller")?,
        microsoft_motion_profile: instance
            .string_to_path("/interaction_profiles/microsoft/motion_controller")?,
        generic_controller_profile: instance
            .string_to_path("/interaction_profiles/khr/generic_controller")?,
        simple_controller_profile: instance
            .string_to_path("/interaction_profiles/khr/simple_controller")?,
        pico4_controller_profile: instance
            .string_to_path("/interaction_profiles/bytedance/pico4_controller")?,
        pico_neo3_controller_profile: instance
            .string_to_path("/interaction_profiles/bytedance/pico_neo3_controller")?,
    })
}

use super::bindings::BindingPaths;

/// All `/user/hand/*/input/...` subpaths used when building [`BindingPaths`].
pub(super) fn resolve_binding_subpaths(
    instance: &xr::Instance,
) -> Result<BindingPaths, xr::sys::Result> {
    let left_grip_pose_path = instance.string_to_path("/user/hand/left/input/grip/pose")?;
    let right_grip_pose_path = instance.string_to_path("/user/hand/right/input/grip/pose")?;
    let left_aim_pose_path = instance.string_to_path("/user/hand/left/input/aim/pose")?;
    let right_aim_pose_path = instance.string_to_path("/user/hand/right/input/aim/pose")?;
    let left_trigger_value_path = instance.string_to_path("/user/hand/left/input/trigger/value")?;
    let right_trigger_value_path =
        instance.string_to_path("/user/hand/right/input/trigger/value")?;
    let left_trigger_touch_path = instance.string_to_path("/user/hand/left/input/trigger/touch")?;
    let right_trigger_touch_path =
        instance.string_to_path("/user/hand/right/input/trigger/touch")?;
    let left_trigger_click_path = instance.string_to_path("/user/hand/left/input/trigger/click")?;
    let right_trigger_click_path =
        instance.string_to_path("/user/hand/right/input/trigger/click")?;
    let left_squeeze_value_path = instance.string_to_path("/user/hand/left/input/squeeze/value")?;
    let right_squeeze_value_path =
        instance.string_to_path("/user/hand/right/input/squeeze/value")?;
    let left_squeeze_click_path = instance.string_to_path("/user/hand/left/input/squeeze/click")?;
    let right_squeeze_click_path =
        instance.string_to_path("/user/hand/right/input/squeeze/click")?;
    let left_thumbstick_path = instance.string_to_path("/user/hand/left/input/thumbstick")?;
    let right_thumbstick_path = instance.string_to_path("/user/hand/right/input/thumbstick")?;
    let left_thumbstick_touch_path =
        instance.string_to_path("/user/hand/left/input/thumbstick/touch")?;
    let right_thumbstick_touch_path =
        instance.string_to_path("/user/hand/right/input/thumbstick/touch")?;
    let left_thumbstick_click_path =
        instance.string_to_path("/user/hand/left/input/thumbstick/click")?;
    let right_thumbstick_click_path =
        instance.string_to_path("/user/hand/right/input/thumbstick/click")?;
    let left_trackpad_path = instance.string_to_path("/user/hand/left/input/trackpad")?;
    let right_trackpad_path = instance.string_to_path("/user/hand/right/input/trackpad")?;
    let left_trackpad_touch_path =
        instance.string_to_path("/user/hand/left/input/trackpad/touch")?;
    let right_trackpad_touch_path =
        instance.string_to_path("/user/hand/right/input/trackpad/touch")?;
    let left_trackpad_click_path =
        instance.string_to_path("/user/hand/left/input/trackpad/click")?;
    let right_trackpad_click_path =
        instance.string_to_path("/user/hand/right/input/trackpad/click")?;
    let left_trackpad_force_path =
        instance.string_to_path("/user/hand/left/input/trackpad/force")?;
    let right_trackpad_force_path =
        instance.string_to_path("/user/hand/right/input/trackpad/force")?;
    let left_x_click_path = instance.string_to_path("/user/hand/left/input/x/click")?;
    let left_y_click_path = instance.string_to_path("/user/hand/left/input/y/click")?;
    let left_x_touch_path = instance.string_to_path("/user/hand/left/input/x/touch")?;
    let left_y_touch_path = instance.string_to_path("/user/hand/left/input/y/touch")?;
    let left_a_click_path = instance.string_to_path("/user/hand/left/input/a/click")?;
    let left_b_click_path = instance.string_to_path("/user/hand/left/input/b/click")?;
    let left_a_touch_path = instance.string_to_path("/user/hand/left/input/a/touch")?;
    let left_b_touch_path = instance.string_to_path("/user/hand/left/input/b/touch")?;
    let right_a_click_path = instance.string_to_path("/user/hand/right/input/a/click")?;
    let right_b_click_path = instance.string_to_path("/user/hand/right/input/b/click")?;
    let right_a_touch_path = instance.string_to_path("/user/hand/right/input/a/touch")?;
    let right_b_touch_path = instance.string_to_path("/user/hand/right/input/b/touch")?;
    let left_menu_click_path = instance.string_to_path("/user/hand/left/input/menu/click")?;
    let right_menu_click_path = instance.string_to_path("/user/hand/right/input/menu/click")?;
    let left_thumbrest_touch_path =
        instance.string_to_path("/user/hand/left/input/thumbrest/touch")?;
    let right_thumbrest_touch_path =
        instance.string_to_path("/user/hand/right/input/thumbrest/touch")?;
    let left_select_click_path = instance.string_to_path("/user/hand/left/input/select/click")?;
    let right_select_click_path = instance.string_to_path("/user/hand/right/input/select/click")?;

    Ok(BindingPaths {
        left_grip_pose: left_grip_pose_path,
        right_grip_pose: right_grip_pose_path,
        left_aim_pose: left_aim_pose_path,
        right_aim_pose: right_aim_pose_path,
        left_trigger_value: left_trigger_value_path,
        right_trigger_value: right_trigger_value_path,
        left_trigger_touch: left_trigger_touch_path,
        right_trigger_touch: right_trigger_touch_path,
        left_trigger_click: left_trigger_click_path,
        right_trigger_click: right_trigger_click_path,
        left_squeeze_value: left_squeeze_value_path,
        right_squeeze_value: right_squeeze_value_path,
        left_squeeze_click: left_squeeze_click_path,
        right_squeeze_click: right_squeeze_click_path,
        left_thumbstick: left_thumbstick_path,
        right_thumbstick: right_thumbstick_path,
        left_thumbstick_touch: left_thumbstick_touch_path,
        right_thumbstick_touch: right_thumbstick_touch_path,
        left_thumbstick_click: left_thumbstick_click_path,
        right_thumbstick_click: right_thumbstick_click_path,
        left_trackpad: left_trackpad_path,
        right_trackpad: right_trackpad_path,
        left_trackpad_touch: left_trackpad_touch_path,
        right_trackpad_touch: right_trackpad_touch_path,
        left_trackpad_click: left_trackpad_click_path,
        right_trackpad_click: right_trackpad_click_path,
        left_trackpad_force: left_trackpad_force_path,
        right_trackpad_force: right_trackpad_force_path,
        left_x_click: left_x_click_path,
        left_y_click: left_y_click_path,
        left_x_touch: left_x_touch_path,
        left_y_touch: left_y_touch_path,
        left_a_click: left_a_click_path,
        left_b_click: left_b_click_path,
        left_a_touch: left_a_touch_path,
        left_b_touch: left_b_touch_path,
        right_a_click: right_a_click_path,
        right_b_click: right_b_click_path,
        right_a_touch: right_a_touch_path,
        right_b_touch: right_b_touch_path,
        left_menu_click: left_menu_click_path,
        right_menu_click: right_menu_click_path,
        left_thumbrest_touch: left_thumbrest_touch_path,
        right_thumbrest_touch: right_thumbrest_touch_path,
        left_select_click: left_select_click_path,
        right_select_click: right_select_click_path,
    })
}
