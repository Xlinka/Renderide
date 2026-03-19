//! Lights buffer command handlers: lights_buffer_renderer_submission.
//!
//! Parses LightsBufferRendererSubmission, reads LightData array from shared memory,
//! and stores in the scene graph's light cache.

use crate::shared::{LightData, RendererCommand};

use super::{CommandContext, CommandHandler, CommandResult};

/// Handles `lights_buffer_renderer_submission`. Reads light data from shared memory
/// and stores it in the scene graph's light cache.
pub struct LightsBufferCommandHandler;

impl CommandHandler for LightsBufferCommandHandler {
    fn handle(&mut self, cmd: &RendererCommand, ctx: &mut CommandContext<'_>) -> CommandResult {
        let RendererCommand::lights_buffer_renderer_submission(data) = cmd else {
            return CommandResult::Ignored;
        };

        if data.lights.is_empty() || data.lights_count <= 0 {
            ctx.scene_graph.light_cache.store_full(data.lights_buffer_unique_id, Vec::new());
            return CommandResult::Handled;
        }

        let Some(shm) = ctx.assets.shared_memory.as_mut() else {
            logger::warn!(
                "LightsBufferRendererSubmission: no shared memory (buffer_id={})",
                data.lights_buffer_unique_id
            );
            return CommandResult::Handled;
        };

        match shm.access_copy_diagnostic_with_context::<LightData>(
            &data.lights,
            Some("LightsBufferRendererSubmission"),
        ) {
            Ok(lights) => {
                ctx.scene_graph
                    .light_cache
                    .store_full(data.lights_buffer_unique_id, lights);
            }
            Err(e) => {
                logger::warn!("LightsBufferRendererSubmission: {}", e);
            }
        }

        CommandResult::Handled
    }
}
