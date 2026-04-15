//! OpenXR action creation, interaction profile paths, and grip/aim space setup.
//!
//! Extracted from [`super::OpenxrInput::new`] to keep the main input type focused on per-frame sampling.

use std::sync::atomic::AtomicU8;

use openxr as xr;

use super::bindings::{apply_suggested_interaction_bindings, ActionRefs, InteractionProfilePaths};
use super::openxr_action_paths::{resolve_binding_subpaths, resolve_user_and_profile_paths};

/// Intermediate container for all actions and spaces produced during [`create_openxr_input_parts`].
pub(super) struct OpenxrInputParts {
    pub action_set: xr::ActionSet,
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
    pub left_profile_cache: AtomicU8,
    pub right_profile_cache: AtomicU8,
    pub left_grip_pose: xr::Action<xr::Posef>,
    pub right_grip_pose: xr::Action<xr::Posef>,
    pub left_trigger: xr::Action<f32>,
    pub right_trigger: xr::Action<f32>,
    pub left_trigger_touch: xr::Action<bool>,
    pub right_trigger_touch: xr::Action<bool>,
    pub left_trigger_click: xr::Action<bool>,
    pub right_trigger_click: xr::Action<bool>,
    pub left_squeeze: xr::Action<f32>,
    pub right_squeeze: xr::Action<f32>,
    pub left_squeeze_click: xr::Action<bool>,
    pub right_squeeze_click: xr::Action<bool>,
    pub left_thumbstick: xr::Action<xr::Vector2f>,
    pub right_thumbstick: xr::Action<xr::Vector2f>,
    pub left_thumbstick_touch: xr::Action<bool>,
    pub right_thumbstick_touch: xr::Action<bool>,
    pub left_thumbstick_click: xr::Action<bool>,
    pub right_thumbstick_click: xr::Action<bool>,
    pub left_trackpad: xr::Action<xr::Vector2f>,
    pub right_trackpad: xr::Action<xr::Vector2f>,
    pub left_trackpad_touch: xr::Action<bool>,
    pub right_trackpad_touch: xr::Action<bool>,
    pub left_trackpad_click: xr::Action<bool>,
    pub right_trackpad_click: xr::Action<bool>,
    pub left_trackpad_force: xr::Action<f32>,
    pub right_trackpad_force: xr::Action<f32>,
    pub left_primary: xr::Action<bool>,
    pub right_primary: xr::Action<bool>,
    pub left_secondary: xr::Action<bool>,
    pub right_secondary: xr::Action<bool>,
    pub left_primary_touch: xr::Action<bool>,
    pub right_primary_touch: xr::Action<bool>,
    pub left_secondary_touch: xr::Action<bool>,
    pub right_secondary_touch: xr::Action<bool>,
    pub left_menu: xr::Action<bool>,
    pub right_menu: xr::Action<bool>,
    pub left_thumbrest_touch: xr::Action<bool>,
    pub right_thumbrest_touch: xr::Action<bool>,
    pub left_select: xr::Action<bool>,
    pub right_select: xr::Action<bool>,
    pub left_space: xr::Space,
    pub right_space: xr::Space,
    pub left_aim_space: xr::Space,
    pub right_aim_space: xr::Space,
}

/// Creates the action set, suggests bindings for known interaction profiles, and builds grip/aim spaces.
///
/// `runtime_supports_generic_controller` must match whether the OpenXR instance was created with
/// `XR_KHR_generic_controller` enabled; when `false`, generic controller binding suggestions are skipped.
///
/// `runtime_supports_bd_controller` must match whether `XR_BD_controller_interaction` was enabled
/// on the instance; when `false`, ByteDance Pico profile binding suggestions are skipped.
pub(super) fn create_openxr_input_parts(
    instance: &xr::Instance,
    session: &xr::Session<xr::Vulkan>,
    runtime_supports_generic_controller: bool,
    runtime_supports_bd_controller: bool,
) -> Result<OpenxrInputParts, xr::sys::Result> {
    let action_set = instance.create_action_set("renderide_input", "Renderide VR input", 0)?;
    let paths_bundle = resolve_user_and_profile_paths(instance)?;
    let binding_paths = resolve_binding_subpaths(instance)?;
    let left_grip_pose =
        action_set.create_action::<xr::Posef>("left_grip_pose", "Left grip pose", &[])?;
    let right_grip_pose =
        action_set.create_action::<xr::Posef>("right_grip_pose", "Right grip pose", &[])?;
    let left_trigger = action_set.create_action::<f32>("left_trigger", "Left trigger", &[])?;
    let right_trigger = action_set.create_action::<f32>("right_trigger", "Right trigger", &[])?;
    let left_trigger_touch =
        action_set.create_action::<bool>("left_trigger_touch", "Left trigger touch", &[])?;
    let right_trigger_touch =
        action_set.create_action::<bool>("right_trigger_touch", "Right trigger touch", &[])?;
    let left_trigger_click =
        action_set.create_action::<bool>("left_trigger_click", "Left trigger click", &[])?;
    let right_trigger_click =
        action_set.create_action::<bool>("right_trigger_click", "Right trigger click", &[])?;
    let left_squeeze = action_set.create_action::<f32>("left_squeeze", "Left squeeze", &[])?;
    let right_squeeze = action_set.create_action::<f32>("right_squeeze", "Right squeeze", &[])?;
    let left_squeeze_click =
        action_set.create_action::<bool>("left_squeeze_click", "Left squeeze click", &[])?;
    let right_squeeze_click =
        action_set.create_action::<bool>("right_squeeze_click", "Right squeeze click", &[])?;
    let left_thumbstick =
        action_set.create_action::<xr::Vector2f>("left_thumbstick", "Left thumbstick", &[])?;
    let right_thumbstick =
        action_set.create_action::<xr::Vector2f>("right_thumbstick", "Right thumbstick", &[])?;
    let left_thumbstick_touch =
        action_set.create_action::<bool>("left_thumbstick_touch", "Left thumbstick touch", &[])?;
    let right_thumbstick_touch = action_set.create_action::<bool>(
        "right_thumbstick_touch",
        "Right thumbstick touch",
        &[],
    )?;
    let left_thumbstick_click =
        action_set.create_action::<bool>("left_thumbstick_click", "Left thumbstick click", &[])?;
    let right_thumbstick_click = action_set.create_action::<bool>(
        "right_thumbstick_click",
        "Right thumbstick click",
        &[],
    )?;
    let left_trackpad =
        action_set.create_action::<xr::Vector2f>("left_trackpad", "Left trackpad", &[])?;
    let right_trackpad =
        action_set.create_action::<xr::Vector2f>("right_trackpad", "Right trackpad", &[])?;
    let left_trackpad_touch =
        action_set.create_action::<bool>("left_trackpad_touch", "Left trackpad touch", &[])?;
    let right_trackpad_touch =
        action_set.create_action::<bool>("right_trackpad_touch", "Right trackpad touch", &[])?;
    let left_trackpad_click =
        action_set.create_action::<bool>("left_trackpad_click", "Left trackpad click", &[])?;
    let right_trackpad_click =
        action_set.create_action::<bool>("right_trackpad_click", "Right trackpad click", &[])?;
    let left_trackpad_force =
        action_set.create_action::<f32>("left_trackpad_force", "Left trackpad force", &[])?;
    let right_trackpad_force =
        action_set.create_action::<f32>("right_trackpad_force", "Right trackpad force", &[])?;
    let left_primary =
        action_set.create_action::<bool>("left_primary", "Left primary button", &[])?;
    let right_primary =
        action_set.create_action::<bool>("right_primary", "Right primary button", &[])?;
    let left_secondary =
        action_set.create_action::<bool>("left_secondary", "Left secondary button", &[])?;
    let right_secondary =
        action_set.create_action::<bool>("right_secondary", "Right secondary button", &[])?;
    let left_primary_touch =
        action_set.create_action::<bool>("left_primary_touch", "Left primary touch", &[])?;
    let right_primary_touch =
        action_set.create_action::<bool>("right_primary_touch", "Right primary touch", &[])?;
    let left_secondary_touch =
        action_set.create_action::<bool>("left_secondary_touch", "Left secondary touch", &[])?;
    let right_secondary_touch =
        action_set.create_action::<bool>("right_secondary_touch", "Right secondary touch", &[])?;
    let left_menu = action_set.create_action::<bool>("left_menu", "Left menu", &[])?;
    let right_menu = action_set.create_action::<bool>("right_menu", "Right menu", &[])?;
    let left_thumbrest_touch =
        action_set.create_action::<bool>("left_thumbrest_touch", "Left thumbrest touch", &[])?;
    let right_thumbrest_touch =
        action_set.create_action::<bool>("right_thumbrest_touch", "Right thumbrest touch", &[])?;
    let left_select = action_set.create_action::<bool>("left_select", "Left select", &[])?;
    let right_select = action_set.create_action::<bool>("right_select", "Right select", &[])?;
    let left_aim_pose =
        action_set.create_action::<xr::Posef>("left_aim_pose", "Left aim pose", &[])?;
    let right_aim_pose =
        action_set.create_action::<xr::Posef>("right_aim_pose", "Right aim pose", &[])?;

    let interaction_profiles = InteractionProfilePaths {
        oculus_touch: paths_bundle.oculus_touch_profile,
        valve_index: paths_bundle.valve_index_profile,
        htc_vive: paths_bundle.htc_vive_profile,
        microsoft_motion: paths_bundle.microsoft_motion_profile,
        generic_controller: paths_bundle.generic_controller_profile,
        simple_controller: paths_bundle.simple_controller_profile,
        pico4_controller: paths_bundle.pico4_controller_profile,
    };

    let action_refs = ActionRefs {
        left_grip_pose: &left_grip_pose,
        right_grip_pose: &right_grip_pose,
        left_aim_pose: &left_aim_pose,
        right_aim_pose: &right_aim_pose,
        left_trigger: &left_trigger,
        right_trigger: &right_trigger,
        left_trigger_touch: &left_trigger_touch,
        right_trigger_touch: &right_trigger_touch,
        left_trigger_click: &left_trigger_click,
        right_trigger_click: &right_trigger_click,
        left_squeeze: &left_squeeze,
        right_squeeze: &right_squeeze,
        left_squeeze_click: &left_squeeze_click,
        right_squeeze_click: &right_squeeze_click,
        left_thumbstick: &left_thumbstick,
        right_thumbstick: &right_thumbstick,
        left_thumbstick_touch: &left_thumbstick_touch,
        right_thumbstick_touch: &right_thumbstick_touch,
        left_thumbstick_click: &left_thumbstick_click,
        right_thumbstick_click: &right_thumbstick_click,
        left_trackpad: &left_trackpad,
        right_trackpad: &right_trackpad,
        left_trackpad_touch: &left_trackpad_touch,
        right_trackpad_touch: &right_trackpad_touch,
        left_trackpad_click: &left_trackpad_click,
        right_trackpad_click: &right_trackpad_click,
        left_trackpad_force: &left_trackpad_force,
        right_trackpad_force: &right_trackpad_force,
        left_primary: &left_primary,
        right_primary: &right_primary,
        left_secondary: &left_secondary,
        right_secondary: &right_secondary,
        left_primary_touch: &left_primary_touch,
        right_primary_touch: &right_primary_touch,
        left_secondary_touch: &left_secondary_touch,
        right_secondary_touch: &right_secondary_touch,
        left_menu: &left_menu,
        right_menu: &right_menu,
        left_thumbrest_touch: &left_thumbrest_touch,
        right_thumbrest_touch: &right_thumbrest_touch,
        left_select: &left_select,
        right_select: &right_select,
    };

    apply_suggested_interaction_bindings(
        instance,
        &interaction_profiles,
        &binding_paths,
        &action_refs,
        runtime_supports_generic_controller,
        runtime_supports_bd_controller,
    )?;

    session.attach_action_sets(&[&action_set])?;
    let left_space = left_grip_pose.create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)?;
    let right_space = right_grip_pose.create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)?;
    let left_aim_space =
        left_aim_pose.create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)?;
    let right_aim_space =
        right_aim_pose.create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)?;
    Ok(OpenxrInputParts {
        action_set,
        left_user_path: paths_bundle.left_user_path,
        right_user_path: paths_bundle.right_user_path,
        oculus_touch_profile: paths_bundle.oculus_touch_profile,
        valve_index_profile: paths_bundle.valve_index_profile,
        htc_vive_profile: paths_bundle.htc_vive_profile,
        microsoft_motion_profile: paths_bundle.microsoft_motion_profile,
        generic_controller_profile: paths_bundle.generic_controller_profile,
        simple_controller_profile: paths_bundle.simple_controller_profile,
        pico4_controller_profile: paths_bundle.pico4_controller_profile,
        pico_neo3_controller_profile: paths_bundle.pico_neo3_controller_profile,
        left_profile_cache: AtomicU8::new(0),
        right_profile_cache: AtomicU8::new(0),
        left_grip_pose,
        right_grip_pose,
        left_trigger,
        right_trigger,
        left_trigger_touch,
        right_trigger_touch,
        left_trigger_click,
        right_trigger_click,
        left_squeeze,
        right_squeeze,
        left_squeeze_click,
        right_squeeze_click,
        left_thumbstick,
        right_thumbstick,
        left_thumbstick_touch,
        right_thumbstick_touch,
        left_thumbstick_click,
        right_thumbstick_click,
        left_trackpad,
        right_trackpad,
        left_trackpad_touch,
        right_trackpad_touch,
        left_trackpad_click,
        right_trackpad_click,
        left_trackpad_force,
        right_trackpad_force,
        left_primary,
        right_primary,
        left_secondary,
        right_secondary,
        left_primary_touch,
        right_primary_touch,
        left_secondary_touch,
        right_secondary_touch,
        left_menu,
        right_menu,
        left_thumbrest_touch,
        right_thumbrest_touch,
        left_select,
        right_select,
        left_space,
        right_space,
        left_aim_space,
        right_aim_space,
    })
}
