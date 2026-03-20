//! On-screen debug HUD rendered with ImGui.

use std::time::{Duration, Instant};

use imgui::{Condition, Context, FontConfig, FontSource, WindowFlags};
use imgui_wgpu::{Renderer, RendererConfig};

use crate::render::RenderTarget;
use crate::render::pass::{MeshDrawPrepStats, SkinnedDebugSample};

/// Per-frame diagnostics sample shown in the debug HUD.
#[derive(Clone, Debug)]
pub struct LiveFrameDiagnostics {
    pub frame_index: i32,
    pub viewport: (u32, u32),
    pub session_update_us: u64,
    pub collect_us: u64,
    pub render_us: u64,
    pub present_us: u64,
    pub total_us: u64,
    pub gpu_mesh_pass_ms: Option<f64>,
    pub batch_count: usize,
    pub overlay_batch_count: usize,
    pub total_draws_in_batches: usize,
    pub overlay_draws_in_batches: usize,
    pub prep_stats: MeshDrawPrepStats,
    pub skinned_samples: Vec<SkinnedDebugSample>,
    pub mesh_cache_count: usize,
    pub pending_render_tasks: usize,
    pub pending_camera_task_readbacks: usize,
    pub frustum_culling_enabled: bool,
    pub rtao_enabled: bool,
    pub ray_tracing_available: bool,
}

impl LiveFrameDiagnostics {
    fn frame_time_ms(&self) -> f64 {
        self.total_us as f64 / 1000.0
    }

    fn fps(&self) -> f64 {
        if self.total_us == 0 {
            0.0
        } else {
            1_000_000.0 / self.total_us as f64
        }
    }

    fn bottleneck(&self) -> &'static str {
        match self.gpu_mesh_pass_ms {
            Some(gpu_ms) if gpu_ms > self.frame_time_ms() => "GPU",
            Some(_) => "CPU",
            None => "CPU?",
        }
    }

    fn submitted_overlay_draws(&self) -> usize {
        self.prep_stats
            .submitted_draws()
            .min(self.total_draws_in_batches)
            .saturating_sub(self.submitted_main_draws())
    }

    fn submitted_main_draws(&self) -> usize {
        let main_draws = self
            .total_draws_in_batches
            .saturating_sub(self.overlay_draws_in_batches);
        self.prep_stats.submitted_draws().min(main_draws)
    }
}

/// ImGui-backed on-screen diagnostics panel.
pub struct DebugHud {
    imgui: Context,
    renderer: Renderer,
    last_frame_at: Instant,
    latest: Option<LiveFrameDiagnostics>,
}

impl DebugHud {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
    ) -> Result<Self, String> {
        let mut imgui = Context::create();
        imgui.set_ini_filename(None);
        imgui.set_log_filename(None);
        imgui.fonts().add_font(&[FontSource::DefaultFontData {
            config: Some(FontConfig {
                oversample_h: 2,
                pixel_snap_h: true,
                size_pixels: 14.0,
                ..FontConfig::default()
            }),
        }]);

        let renderer_config = RendererConfig {
            texture_format: surface_format,
            ..RendererConfig::default()
        };
        let renderer = Renderer::new(&mut imgui, device, queue, renderer_config);

        Ok(Self {
            imgui,
            renderer,
            last_frame_at: Instant::now(),
            latest: None,
        })
    }

    pub fn update(&mut self, sample: LiveFrameDiagnostics) {
        self.latest = Some(sample);
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &RenderTarget,
    ) -> Result<(), String> {
        let (width, height) = target.dimensions();
        let delta = self.last_frame_at.elapsed().max(Duration::from_millis(1));
        self.last_frame_at = Instant::now();

        let io = self.imgui.io_mut();
        io.display_size = [width as f32, height as f32];
        io.update_delta_time(delta);

        let ui = self.imgui.frame();
        let panel_width = 700.0f32;
        let panel_x = (width as f32 - panel_width - 12.0).max(12.0);
        let window_flags = WindowFlags::NO_DECORATION
            | WindowFlags::NO_MOVE
            | WindowFlags::NO_RESIZE
            | WindowFlags::NO_SAVED_SETTINGS
            | WindowFlags::NO_FOCUS_ON_APPEARING
            | WindowFlags::NO_NAV
            | WindowFlags::NO_INPUTS;

        ui.window("Render Debug")
            .position([panel_x, 12.0], Condition::Always)
            .size([panel_width, 0.0], Condition::Always)
            .bg_alpha(0.72)
            .flags(window_flags)
            .build(|| {
                if let Some(sample) = self.latest.as_ref() {
                    ui.text(format!(
                        "FPS {:.1}  |  {:.2} ms  |  {}",
                        sample.fps(),
                        sample.frame_time_ms(),
                        sample.bottleneck()
                    ));
                    ui.text(format!(
                        "Frame {}  |  {}x{}",
                        sample.frame_index, sample.viewport.0, sample.viewport.1
                    ));
                    ui.separator();
                    ui.text(format!(
                        "CPU update {:.2}  collect {:.2}  render {:.2}  present {:.2}",
                        sample.session_update_us as f64 / 1000.0,
                        sample.collect_us as f64 / 1000.0,
                        sample.render_us as f64 / 1000.0,
                        sample.present_us as f64 / 1000.0
                    ));
                    ui.text(match sample.gpu_mesh_pass_ms {
                        Some(ms) => format!("GPU mesh pass {:.2} ms", ms),
                        None => "GPU mesh pass pending".to_string(),
                    });
                    ui.separator();
                    ui.text(format!(
                        "Batches {} total  |  {} main  |  {} overlay",
                        sample.batch_count,
                        sample
                            .batch_count
                            .saturating_sub(sample.overlay_batch_count),
                        sample.overlay_batch_count
                    ));
                    ui.text(format!(
                        "Draws {} total  |  {} main  |  {} overlay",
                        sample.total_draws_in_batches,
                        sample
                            .total_draws_in_batches
                            .saturating_sub(sample.overlay_draws_in_batches),
                        sample.overlay_draws_in_batches
                    ));
                    ui.text(format!(
                        "Submitted {} total  |  {} main  |  {} overlay",
                        sample.prep_stats.submitted_draws(),
                        sample.submitted_main_draws(),
                        sample.submitted_overlay_draws()
                    ));
                    ui.separator();
                    ui.text(format!(
                        "Prep rigid {}  skinned {}",
                        sample.prep_stats.rigid_input_draws, sample.prep_stats.skinned_input_draws
                    ));
                    ui.text(format!(
                        "Culled rigid {}  |  degenerate skip {}",
                        sample.prep_stats.frustum_culled_rigid_draws,
                        sample.prep_stats.skipped_cull_degenerate_bounds
                    ));
                    ui.text(format!(
                        "Missing mesh {}  empty mesh {}  missing GPU {}",
                        sample.prep_stats.skipped_missing_mesh_asset,
                        sample.prep_stats.skipped_empty_mesh,
                        sample.prep_stats.skipped_missing_gpu_buffers
                    ));
                    ui.text(format!(
                        "Skinned skips bind {}  ids {}  mismatch {}  vb {}",
                        sample.prep_stats.skipped_skinned_missing_bind_poses,
                        sample.prep_stats.skipped_skinned_missing_bone_ids,
                        sample.prep_stats.skipped_skinned_id_count_mismatch,
                        sample.prep_stats.skipped_skinned_missing_vertex_buffer
                    ));
                    ui.separator();
                    ui.text(format!(
                        "Mesh cache {}  |  tasks {}  |  readbacks {}",
                        sample.mesh_cache_count,
                        sample.pending_render_tasks,
                        sample.pending_camera_task_readbacks
                    ));
                    ui.text(format!(
                        "Flags cull={}  rtao={}  raytracing={}",
                        sample.frustum_culling_enabled,
                        sample.rtao_enabled,
                        sample.ray_tracing_available
                    ));
                } else {
                    ui.text("Waiting for frame diagnostics...");
                }
            });

        let draw_data = self.imgui.render();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("imgui debug hud encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("imgui debug hud pass"),
                timestamp_writes: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target.color_view(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.renderer
                .render(draw_data, queue, device, &mut pass)
                .map_err(|e| format!("imgui render failed: {e}"))?;
        }
        queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(total_us: u64, gpu_ms: Option<f64>) -> LiveFrameDiagnostics {
        LiveFrameDiagnostics {
            frame_index: 12,
            viewport: (1280, 720),
            session_update_us: 1_000,
            collect_us: 2_000,
            render_us: 3_000,
            present_us: 500,
            total_us,
            gpu_mesh_pass_ms: gpu_ms,
            batch_count: 4,
            overlay_batch_count: 1,
            total_draws_in_batches: 20,
            overlay_draws_in_batches: 5,
            prep_stats: MeshDrawPrepStats {
                rigid_input_draws: 12,
                skinned_input_draws: 8,
                submitted_rigid_draws: 10,
                submitted_skinned_draws: 8,
                ..MeshDrawPrepStats::default()
            },
            skinned_samples: vec![SkinnedDebugSample {
                space_id: 7,
                node_id: 9,
                mesh_asset_id: 12,
                is_overlay: false,
                vertex_count: 1024,
                bind_pose_count: 32,
                bone_ids_len: 32,
                root_bone_transform_id: Some(3),
                model_position: [1.0, 2.0, 3.0],
                root_bone_world_position: Some([0.5, 1.5, 2.5]),
                v0_bone_info: vec![
                    (2, 3, Some([0.5, 1.5, 2.5])),
                    (17, 4, Some([0.6, 1.6, 2.6])),
                    (3, 5, None),
                ],
                first_vertex_indices: [0, 1, 2, 3],
                first_vertex_weights: [0.4, 0.3, 0.2, 0.1],
                blendshape_weights_preview: vec![1.0, 0.5],
                all_bone_slots: vec![
                    BoneSlotInfo { tid: 3, world_pos: Some([0.5, 1.5, 2.5]), parent_tid: -1, parent_world_pos: None },
                    BoneSlotInfo { tid: 4, world_pos: Some([0.5, 0.1, 2.5]), parent_tid: 3, parent_world_pos: Some([0.5, 1.5, 2.5]) },
                ],
                bad_bone_slots: vec![1],
                root_chain: vec![(3, Some([0.5, 1.5, 2.5])), (1, Some([0.0, 0.0, 0.0]))],
            }],
            mesh_cache_count: 10,
            pending_render_tasks: 0,
            pending_camera_task_readbacks: 0,
            frustum_culling_enabled: true,
            rtao_enabled: true,
            ray_tracing_available: true,
        }
    }

    #[test]
    fn bottleneck_prefers_gpu_when_gpu_time_exceeds_cpu_frame_time() {
        assert_eq!(sample(4_000, Some(8.0)).bottleneck(), "GPU");
        assert_eq!(sample(12_000, Some(4.0)).bottleneck(), "CPU");
    }

    #[test]
    fn submitted_overlay_draws_never_underflow() {
        let sample = sample(10_000, None);
        assert_eq!(sample.submitted_main_draws(), 15);
        assert_eq!(sample.submitted_overlay_draws(), 3);
    }
}
