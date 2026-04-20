A modern Rust + wgpu renderer for Resonite.

## Status

Experimental: Performance, stability, and platform support are still evolving.  
Visual bugs and missing features are expected.

## Building and Running

1. Ensure you have a Steam installation of [Resonite](https://store.steampowered.com/app/2519830/Resonite/).

1. Clone this repository and switch to the `Renderide/` directory:

   ```bash
   git clone https://github.com/DoubleStyx/Renderide.git
   cd Renderide
   ```

1. Install Rust with [Rustup](https://rustup.rs/) (if missing) and build the renderer:

   ```bash
   cargo build --release
   ```

1. Run the bootstrapper:

   ```bash
   ./target/release/bootstrapper
   ```

The bootstrapper will launch the Resonite host and connect Renderide automatically.

## Debugging

1. Build the workspace in dev mode:

   ```bash
   cargo build --profile dev-fast
   ```

1. Run the bootstrapper in dev mode:

   ```bash
   RUST_BACKTRACE=1 ./target/dev-fast/bootstrapper
   ```

1. Enable validation layers in the config hud to get more detailed error messages for GPU crashes. Requires a restart.

1. Inspect logs in the `logs/` folder for panics, crashes, backtraces, and validation errors.

## Profiling

Renderide integrates with [Tracy](https://github.com/wolfpld/tracy) for CPU and GPU profiling.
CPU spans come from the `profiling` crate; GPU timestamp queries come from `wgpu-profiler`.
GPU timing requires `TIMESTAMP_QUERY` and `TIMESTAMP_QUERY_INSIDE_ENCODERS` adapter support.
If either is missing, a warning is logged and only CPU spans are emitted.

### Building with profiling enabled

```bash
cargo build --profile dev-fast --features tracy
```
```

### Connecting Tracy

1. Download the Tracy profiler GUI from the [Tracy releases page](https://github.com/wolfpld/tracy/releases)
   and launch it.

1. Start Renderide normally (bootstrapper or renderer directly).

1. In the Tracy GUI, connect to `localhost` on port **8086**.

Renderide uses Tracy's `ondemand` mode: data is only streamed while the GUI is connected, so
profiled builds carry near-zero runtime cost when Tracy is not attached.