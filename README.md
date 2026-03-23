# Renderide

A Rust renderer for Resonite, replacing the Unity renderer with a custom Unity-like one using [wgpu](https://github.com/gfx-rs/wgpu).

## Warning

This renderer is experimental: performance, platform support, and stability are limited. It is not for general consumer use currently; many features are enabled for testing, and visual bugs or unexpected behavior are possible.

## Prerequisites

### Rust

Install the current stable toolchain with [rustup](https://rustup.rs/). On Linux and macOS you can use:

`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

On Windows, download and run [rustup-init.exe](https://rustup.rs/) from the same site, or use `winget install Rustlang.Rustup` if you use winget.

This workspace uses Rust edition 2024. If `cargo build` fails with an edition error, run `rustup update stable` and try again.

### Resonite and Renderite

To run the full stack you need a Resonite installation that includes Renderite (so `Renderite.Host.dll` is present). The bootstrapper runs the host with the system `dotnet` executable; install a compatible .NET runtime if prompted.

Resonite is located via `RESONITE_DIR`, `STEAM_PATH`, common Steam install paths, or Steam’s `libraryfolders.vdf`. On Windows the bootstrapper can also consult the Steam registry. Set `RESONITE_DIR` to your game folder if discovery fails.

### Optional (contributors)

- [.NET 10 SDK](https://dotnet.microsoft.com/download) for SharedTypeGenerator and UnityShaderConverter under `generators/`.
- [Slang](https://shader-slang.com/) only if you run UnityShaderConverter with `slangc` (put `slangc` on `PATH`, set `SLANGC`, or pass `--slangc`). Use `--skip-slang` to skip that when committed WGSL is enough.

## Build and run

Run these from the `Renderide/` directory (the Cargo workspace root).

```bash
cargo build --release
cargo run --release -p bootstrapper
```

The bootstrapper expects the `renderide` binary next to itself (for example under `target/release/`). On Linux it may run the process as `Renderite.Renderer` and create a symlink to `renderide` if needed; on Windows it uses `renderide.exe`.

On Linux, if Wine is in use, the bootstrapper uses `LinuxBootstrap.sh` from the Resonite directory. On Windows, Wine does not apply.

Optional [`configuration.ini`](configuration.ini) may sit next to the renderer executable or in the current working directory. Keys and layout are defined in [`crates/renderide/src/config.rs`](crates/renderide/src/config.rs).

## Overview

Resonite (formerly Neos VR) is a VR and social platform. Renderite is its renderer abstraction (Host, Shared, Unity). Renderide is a cross-platform drop-in renderer that works with the native .NET host on Windows, Linux, and other supported setups.

```mermaid
flowchart TB
    subgraph Bootstrapper [Bootstrapper Rust]
        CreateQueues[Create IPC queues]
        SpawnHost[Spawn Renderite.Host]
        QueueLoop[Queue command loop]
        SpawnRenderer[Spawn Renderide]
    end
    subgraph Host [Renderite.Host C#]
        GameEngine[Game engine init]
    end
    subgraph Renderide [Renderide Rust]
        Renderer[wgpu winit renderer]
    end
    CreateQueues --> SpawnHost
    SpawnHost --> QueueLoop
    QueueLoop -->|renderer CLI args -> spawn| SpawnRenderer
    Host <-->|bootstrapper_* queues| Bootstrapper
    Renderide <-->|session IPC queues| Host
```

## Repository layout

IPC summary: `{prefix}.bootstrapper_in` / `{prefix}.bootstrapper_out` connect the host and bootstrapper. The host and Renderide use separate shared-memory queues named in the renderer argv. Messages other than control tokens such as `HEARTBEAT` and `SHUTDOWN` can carry argv tokens used to spawn `renderide`.

### Rust crates

| Crate | Path | Role |
|-------|------|------|
| interprocess | [`crates/interprocess/`](crates/interprocess/) | Shared-memory IPC queues |
| logger | [`crates/logger/`](crates/logger/) | Logging shared by bootstrapper and renderer |
| bootstrapper | [`crates/bootstrapper/`](crates/bootstrapper/) | Spawns Renderite.Host, queue loop, starts Renderide |
| renderide | [`crates/renderide/`](crates/renderide/) | Renderer (`renderide`, `roundtrip` binaries), scene, assets, shaders |

### Third-party trees

| Path | Role |
|------|------|
| [`third_party/UnityShaderParser/`](third_party/UnityShaderParser/) | Unity ShaderLab / HLSL parsing for UnityShaderConverter |
| [`third_party/Resonite.UnityShaders/`](third_party/Resonite.UnityShaders/) | Upstream Resonite shaders (default converter input roots include this and sample shaders) |

## Development

### SharedTypeGenerator

`generators/SharedTypeGenerator/` (C# / .NET 10) turns `Renderite.Shared.dll` into Rust types and pack logic in `crates/renderide/src/shared/shared.rs` (generated; edit the generator, not that file by hand).

```bash
cd Renderide
dotnet run --project generators/SharedTypeGenerator -- -i /path/to/Renderite.Shared.dll
```

Use `-o` to write elsewhere. For pipeline details, see the project sources.

### UnityShaderConverter

`generators/UnityShaderConverter/` (C# / .NET 10) emits Rust modules and WGSL under `crates/renderide/src/shaders/generated/`. Uses UnityShaderParser. Unrelated to SharedTypeGenerator.

Typical commands (from `Renderide/`):

```bash
# Fast: no slangc; keeps existing WGSL on disk where present
dotnet run --project generators/UnityShaderConverter -- --skip-slang

# Full run including slangc (Slang on PATH, or SLANGC / --slangc)
dotnet run --project generators/UnityShaderConverter --
```

Add `-v` / `--verbose` for more log output. `dotnet run --project generators/UnityShaderConverter -- --help` lists flags such as `--input`, `--output`, `--compiler-config`, and `--variant-config`.

### Debugging

Logs under `logs/` relative to the bootstrapper’s working directory include `Bootstrapper.log` and `HostOutput.log`. The renderer appends `Renderide.log`; its path also follows the renderide crate layout at build time, so the working directory can produce two different `Renderide.log` files. Run from a consistent directory if that matters.

Verbosity: bootstrapper defaults to `trace`; renderer to `info` unless `-LogLevel` is passed. Pass `--log-level <level>` or `-l <level>` on the bootstrapper (`error`, `warn`, `info`, `debug`, `trace`) to cap both and forward `-LogLevel` to Renderide.

```bat
cargo run --release -p bootstrapper -- --log-level debug
```

GPU validation: set `RENDERIDE_GPU_VALIDATION=1` (or `true` or `yes`) before first GPU init to enable validation in [`crates/renderide/src/config.rs`](crates/renderide/src/config.rs). On Linux and macOS you typically export the variable in the shell before starting the process; on Windows use `set RENDERIDE_GPU_VALIDATION=1` in Command Prompt or `$env:RENDERIDE_GPU_VALIDATION=1` in PowerShell for the current session.

wgpu’s `WGPU_VALIDATION` is still honored via [`InstanceFlags::with_env`](https://docs.rs/wgpu/latest/wgpu/struct.InstanceFlags.html#method.with_env). Expect a large performance cost; use only while debugging API issues.

## Goals

- Solid cross-platform renderer with correct materials, meshes, and textures
- Strong typing for host and renderer IPC via generated shared types
- Native .NET host on Windows and Linux, with Linux Wine paths where applicable
- Room to grow toward advanced lighting and effects as the stack matures

## License

See [LICENSE](LICENSE).
