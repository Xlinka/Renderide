//! Frame timing HUD window (wall-clock FPS and submit splits).

use crate::diagnostics::FrameTimingHudSnapshot;
use imgui::{Condition, WindowFlags};

use super::super::fmt as hud_fmt;
use super::super::layout as overlay_layout;

pub(super) fn frame_timing_window(ui: &imgui::Ui, timing: Option<&FrameTimingHudSnapshot>) {
    let window_flags = WindowFlags::ALWAYS_AUTO_RESIZE
        | WindowFlags::NO_SAVED_SETTINGS
        | WindowFlags::NO_FOCUS_ON_APPEARING
        | WindowFlags::NO_NAV;
    ui.window("Frame timing")
        .position(overlay_layout::frame_timing_xy(), Condition::FirstUseEver)
        .bg_alpha(0.72)
        .flags(window_flags)
        .build(|| {
            let Some(t) = timing else {
                ui.text("Waiting for snapshot…");
                return;
            };
            let fps = t.fps_from_wall();
            ui.text(format!("FPS {}", hud_fmt::f64_field(8, 2, fps)));
            ui.text(format!(
                "Frame time (ms) {}",
                hud_fmt::f64_field(8, 3, t.wall_frame_time_ms)
            ));
            if let Some(ms) = t.cpu_frame_until_submit_ms {
                ui.text(format!(
                    "CPU (tick to last submit) {} ms",
                    hud_fmt::f64_field(8, 3, ms)
                ));
            } else {
                ui.text_disabled("CPU (tick to last submit): n/a");
            }
            if let Some(ms) = t.gpu_frame_after_submit_ms {
                ui.text(format!(
                    "GPU (last completed submit→idle) {} ms",
                    hud_fmt::f64_field(8, 3, ms)
                ));
            } else {
                ui.text_disabled("GPU (last completed submit→idle): n/a");
            }
        });
}
