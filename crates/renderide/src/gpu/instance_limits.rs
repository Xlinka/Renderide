//! [`wgpu::InstanceFlags`] and [`wgpu::Limits`] helpers used when creating the GPU instance/device.

/// Requests [`wgpu::Limits`] for [`wgpu::Adapter::request_device`].
///
/// Starts from WebGPU-tier [`wgpu::Limits::default`], raises renderer-critical material binding
/// caps where native adapters expose them, then clamps every field to what the adapter supports via
/// [`wgpu::Limits::or_worse_values_from`] so GPUs with lower caps (for example
/// [`wgpu::Limits::max_color_attachments`] below 8 on some ARM/Mali stacks) do not fail device
/// creation.
///
/// After clamping, [`wgpu::Limits::max_buffer_size`] and
/// [`wgpu::Limits::max_storage_buffer_binding_size`] are set from the adapter so large mesh uploads
/// (blendshape packs, etc.) can use the full reported allowance—WebGPU defaults alone cap
/// [`wgpu::Limits::max_buffer_size`] at 256 MiB while the adapter often allows more.
/// [`wgpu::Limits::max_texture_dimension_2d`] is capped at **16384** when the adapter allows it,
/// matching the host’s maximum 2D texture size.
pub(crate) fn required_limits_for_adapter(adapter: &wgpu::Adapter) -> wgpu::Limits {
    required_limits_from_adapter_limits(adapter.limits())
}

pub(crate) fn required_limits_from_adapter_limits(adapter_limits: wgpu::Limits) -> wgpu::Limits {
    let al = adapter_limits;
    let mut desired = wgpu::Limits::default();
    desired.max_samplers_per_shader_stage = desired.max_samplers_per_shader_stage.max(32);
    desired.max_sampled_textures_per_shader_stage =
        desired.max_sampled_textures_per_shader_stage.max(32);

    let mut limits = desired.or_worse_values_from(&al);
    limits.max_buffer_size = al.max_buffer_size;
    limits.max_storage_buffer_binding_size = al.max_storage_buffer_binding_size;

    limits.max_texture_dimension_2d = std::cmp::min(al.max_texture_dimension_2d, 16384);
    limits
}

/// Base flags from the renderer config (validation), before [`wgpu::InstanceFlags::with_env`].
pub(crate) fn instance_flags_base(gpu_validation_layers: bool) -> wgpu::InstanceFlags {
    let mut flags = wgpu::InstanceFlags::empty();
    if gpu_validation_layers {
        flags.insert(wgpu::InstanceFlags::VALIDATION);
    }
    flags
}

/// Builds [`wgpu::InstanceFlags`] for desktop GPU init: optional `VALIDATION`, then
/// [`wgpu::InstanceFlags::with_env`] so `WGPU_VALIDATION` and related variables can override at
/// process start.
pub fn instance_flags_for_gpu_init(gpu_validation_layers: bool) -> wgpu::InstanceFlags {
    instance_flags_base(gpu_validation_layers).with_env()
}

/// Returns the [`wgpu::Backends`] mask used at instance creation, scoped per-OS to skip backends
/// that wgpu would otherwise probe at startup.
///
/// On Windows the OpenGL/WGL probe alone takes ~370 ms on a stock NVIDIA driver — visible in the
/// Tracy trace as `Init OpenGL (WGL) Backend` blocking the main thread. The renderer never picks
/// the GL backend on Windows (DX12/Vulkan are always preferred), so probing it is pure dead weight
/// at startup. Other platforms keep `Backends::all()` so adapter selection is unchanged there.
pub fn backends_for_gpu_init() -> wgpu::Backends {
    #[cfg(windows)]
    {
        wgpu::Backends::DX12 | wgpu::Backends::VULKAN
    }
    #[cfg(not(windows))]
    {
        wgpu::Backends::all()
    }
}

/// Builds [`wgpu::BackendOptions`] preferring DXC over FXC for DX12 shader compilation.
///
/// `Fxc` is the legacy HLSL compiler and can stall the calling thread for many seconds on
/// permutation-heavy shaders (the new triplanar / displace materials hit this); `DynamicDxc`
/// loads `dxcompiler.dll` next to the executable and compiles dramatically faster. Falls back
/// to wgpu's `Auto` (static → dynamic → FXC chain) when no `dxcompiler.dll` sits next to the
/// running binary so behavior degrades gracefully on machines without DXC.
pub fn backend_options_for_gpu_init() -> wgpu::BackendOptions {
    let mut options = wgpu::BackendOptions::default();
    options.dx12.shader_compiler = select_dx12_shader_compiler();
    options.with_env()
}

#[cfg(windows)]
fn select_dx12_shader_compiler() -> wgpu::Dx12Compiler {
    if let Some(dxc_path) = dxcompiler_dll_next_to_exe() {
        logger::info!(
            "DX12 shader compiler: DynamicDxc (dxcompiler.dll at {})",
            dxc_path.display()
        );
        return wgpu::Dx12Compiler::DynamicDxc {
            dxc_path: dxc_path.to_string_lossy().into_owned(),
        };
    }
    logger::warn!(
        "DX12 shader compiler: dxcompiler.dll not found next to renderide.exe — \
         wgpu will use its default Auto chain (static→dynamic→FXC). On a fresh build \
         this typically falls back to FXC, which can stall the main thread for seconds \
         per shader. Drop dxcompiler.dll into the same folder as renderide.exe for fast \
         compiles (Windows SDK ships it under \
         `C:\\Program Files (x86)\\Windows Kits\\10\\bin\\<sdk_ver>\\x64\\dxcompiler.dll`)."
    );
    wgpu::Dx12Compiler::Auto
}

#[cfg(not(windows))]
fn select_dx12_shader_compiler() -> wgpu::Dx12Compiler {
    wgpu::Dx12Compiler::Auto
}

#[cfg(windows)]
fn dxcompiler_dll_next_to_exe() -> Option<std::path::PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    let candidate = dir.join("dxcompiler.dll");
    if candidate.is_file() {
        Some(candidate)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{instance_flags_base, required_limits_from_adapter_limits};
    use wgpu::InstanceFlags;

    #[test]
    fn instance_flags_base_toggles_validation() {
        assert!(!instance_flags_base(false).contains(InstanceFlags::VALIDATION));
        assert!(instance_flags_base(true).contains(InstanceFlags::VALIDATION));
    }

    #[test]
    fn required_limits_raise_material_texture_budget_when_adapter_allows() {
        let adapter_limits = wgpu::Limits {
            max_samplers_per_shader_stage: 64,
            max_sampled_textures_per_shader_stage: 64,
            ..wgpu::Limits::default()
        };

        let required = required_limits_from_adapter_limits(adapter_limits);

        assert_eq!(required.max_samplers_per_shader_stage, 32);
        assert_eq!(required.max_sampled_textures_per_shader_stage, 32);
    }

    #[test]
    fn required_limits_clamp_material_texture_budget_to_adapter() {
        let adapter_limits = wgpu::Limits {
            max_samplers_per_shader_stage: 18,
            max_sampled_textures_per_shader_stage: 18,
            ..wgpu::Limits::default()
        };

        let required = required_limits_from_adapter_limits(adapter_limits);

        assert_eq!(required.max_samplers_per_shader_stage, 18);
        assert_eq!(required.max_sampled_textures_per_shader_stage, 18);
    }
}
