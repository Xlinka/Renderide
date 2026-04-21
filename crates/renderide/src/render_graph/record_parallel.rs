//! Scaffolding for per-view parallel command encoding under
//! [`crate::config::RecordParallelism::PerViewParallel`].
//!
//! The infrastructure for parallel recording is prepared in Milestones A-D of the Phase 4
//! parallelization effort: `record(&self, …)` on every pass trait, the
//! [`crate::render_graph::FrameSystemsShared`] / [`crate::render_graph::FrameRenderParamsView`]
//! split, [`crate::render_graph::FrameUploadBatch`] draining on the main thread post-scope,
//! transient textures/buffers pre-resolved once before the per-view loop, and per-view scratch
//! slabs keyed by [`crate::backend::OcclusionViewId`].
//!
//! Full rayon fan-out with `rayon::scope` additionally requires the per-view encoder helper to
//! take `&self` and operate on shared (`&`) access to the GPU context and render backend, plus
//! a gate for the singleton `GpuProfiler` take/restore pattern. Until that refactor lands, the
//! parallel branch logs once via [`warn_parallel_falls_back_once`] and falls back to serial.

use std::sync::atomic::{AtomicBool, Ordering};

/// One-time latch so the fallback `info!` only fires on the first frame after opt-in.
static PARALLEL_FALLBACK_LOGGED: AtomicBool = AtomicBool::new(false);

/// Logs a single `info!` the first time per-view parallel recording is requested but the fallback
/// is taken. Subsequent calls are no-ops.
///
/// `view_count` is recorded so the log reflects the VR / secondary-camera context that motivated
/// the opt-in.
pub fn warn_parallel_falls_back_once(view_count: usize) {
    if !PARALLEL_FALLBACK_LOGGED.swap(true, Ordering::Relaxed) {
        logger::info!(
            "record_parallelism = PerViewParallel requested for {view_count} views; \
             Milestone E scaffolding is in place — full rayon fan-out is a follow-up, \
             recording serially this frame."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exercises the one-time log latch so repeated calls do not re-log.
    #[test]
    fn warn_parallel_falls_back_once_latches_after_first_call() {
        PARALLEL_FALLBACK_LOGGED.store(false, Ordering::Relaxed);
        warn_parallel_falls_back_once(2);
        assert!(PARALLEL_FALLBACK_LOGGED.load(Ordering::Relaxed));
        warn_parallel_falls_back_once(4);
        assert!(PARALLEL_FALLBACK_LOGGED.load(Ordering::Relaxed));
    }
}
