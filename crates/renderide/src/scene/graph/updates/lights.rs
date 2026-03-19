//! Lights buffer renderer updates from host.
//!
//! Applies incremental updates (states, removals, additions) to the light cache
//! for each render space.

use crate::ipc::shared_memory::SharedMemoryAccessor;
use crate::scene::LightCache;
use crate::shared::{LightsBufferRendererState, LightsBufferRendererUpdate};

use super::super::error::SceneError;

/// Applies lights buffer renderer updates from host.
///
/// Reads removals (i32 indices), additions (LightsBufferRendererState), and
/// states (LightsBufferRendererState) from shared memory and merges with the
/// light cache. Uses space_id as the buffer key (1:1 mapping with lights_buffer_unique_id).
pub(crate) fn apply_lights_buffer_renderers_update(
    light_cache: &mut LightCache,
    shm: &mut SharedMemoryAccessor,
    update: &LightsBufferRendererUpdate,
    space_id: i32,
) -> Result<(), SceneError> {
    let i32_size = std::mem::size_of::<i32>() as i32;
    let state_size = std::mem::size_of::<LightsBufferRendererState>() as i32;

    let removals = if update.removals.length >= i32_size {
        shm.access_copy_diagnostic::<i32>(&update.removals)
            .map_err(SceneError::SharedMemoryAccess)?
    } else {
        Vec::new()
    };

    let additions = if update.additions.length >= state_size {
        shm.access_copy_diagnostic::<LightsBufferRendererState>(&update.additions)
            .map_err(SceneError::SharedMemoryAccess)?
    } else {
        Vec::new()
    };

    let states = if update.states.length >= state_size {
        shm.access_copy_diagnostic::<LightsBufferRendererState>(&update.states)
            .map_err(SceneError::SharedMemoryAccess)?
    } else {
        Vec::new()
    };

    light_cache.apply_update(space_id, &removals, &additions, &states);

    Ok(())
}
