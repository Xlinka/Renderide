//! Renderer config overlay window (Figment-backed settings).

use crate::config::RendererSettingsHandle;
use imgui::Condition;

use super::super::layout as overlay_layout;
use super::renderer_config::renderer_config_panel_body;

/// Third overlay window: editable [`crate::config::RendererSettings`] with immediate disk sync.
pub(super) fn renderer_config_window(
    ui: &imgui::Ui,
    settings: &RendererSettingsHandle,
    save_path: &std::path::Path,
    suppress_renderer_config_disk_writes: bool,
    open: &mut bool,
) {
    ui.window("Renderer config")
        .opened(open)
        .position(
            [overlay_layout::MARGIN, overlay_layout::MARGIN],
            Condition::FirstUseEver,
        )
        .size(
            [
                overlay_layout::RENDERER_CONFIG_W,
                overlay_layout::RENDERER_CONFIG_H,
            ],
            Condition::FirstUseEver,
        )
        .bg_alpha(0.88)
        .build(|| {
            ui.text_wrapped(
                "This file is owned by the renderer. Do not edit config.toml manually while \
                 the process is running — your changes may be overwritten or lost. Use these \
                 controls instead.",
            );
            if suppress_renderer_config_disk_writes {
                ui.text_colored(
                    [1.0, 0.35, 0.35, 1.0],
                    "Disk save is disabled: startup Figment extract failed. Fix config.toml and restart.",
                );
            }
            ui.separator();

            let Ok(mut g) = settings.write() else {
                ui.text_colored([1.0, 0.4, 0.4, 1.0], "Settings store is unavailable.");
                return;
            };

            renderer_config_panel_body(ui, &mut g, save_path, suppress_renderer_config_disk_writes);
        });
}
