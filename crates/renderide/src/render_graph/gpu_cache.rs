//! Small GPU cache primitives for render-graph effect passes.

use std::hash::Hash;
use std::num::NonZeroU32;
use std::sync::{Arc, OnceLock};

use hashbrown::HashMap;
use parking_lot::Mutex;

/// One-time GPU object slot.
#[derive(Debug)]
pub(crate) struct OnceGpu<T> {
    /// Lazily initialized GPU object.
    slot: OnceLock<T>,
}

impl<T> Default for OnceGpu<T> {
    fn default() -> Self {
        Self {
            slot: OnceLock::new(),
        }
    }
}

impl<T> OnceGpu<T> {
    /// Returns the cached object, creating it with `build` on first use.
    pub(crate) fn get_or_create(&self, build: impl FnOnce() -> T) -> &T {
        self.slot.get_or_init(build)
    }
}

/// Descriptor for a fullscreen triangle render pipeline.
pub(crate) struct FullscreenRenderPipelineDesc<'a> {
    /// Debug label applied to the pipeline layout and render pipeline.
    pub(crate) label: &'a str,
    /// Bind group layouts used by the pipeline layout.
    pub(crate) bind_group_layouts: &'a [Option<&'a wgpu::BindGroupLayout>],
    /// Shader module containing `vs_main` and the selected fragment entry point.
    pub(crate) shader: &'a wgpu::ShaderModule,
    /// Fragment entry point for the pass.
    pub(crate) fragment_entry: &'a str,
    /// Single color attachment format.
    pub(crate) output_format: wgpu::TextureFormat,
    /// Optional color blend state.
    pub(crate) blend: Option<wgpu::BlendState>,
    /// Whether the pipeline records as a two-eye multiview pass.
    pub(crate) multiview_stereo: bool,
}

/// Creates a WGSL shader module with the renderer's standard descriptor shape.
pub(crate) fn create_wgsl_shader_module(
    device: &wgpu::Device,
    label: &str,
    source: &str,
) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

/// Returns the `0b11` multiview mask used for stereo eye layers.
pub(crate) fn stereo_multiview_mask() -> Option<NonZeroU32> {
    NonZeroU32::new(3)
}

/// Returns the stereo multiview mask when `multiview_stereo` is active.
pub(crate) fn multiview_mask(multiview_stereo: bool) -> Option<NonZeroU32> {
    multiview_stereo.then(stereo_multiview_mask).flatten()
}

/// Returns the stereo mask for active multiview, otherwise preserves a template mask.
pub(crate) fn stereo_mask_or_template(
    multiview_stereo: bool,
    template_mask: Option<NonZeroU32>,
) -> Option<NonZeroU32> {
    if multiview_stereo {
        stereo_multiview_mask()
    } else {
        template_mask
    }
}

/// Creates a linear clamp sampler for fullscreen texture sampling.
pub(crate) fn create_linear_clamp_sampler(device: &wgpu::Device, label: &str) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(label),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        ..Default::default()
    })
}

/// Number of array layers to expose when sampling a texture as `texture_2d_array`.
pub(crate) fn d2_array_layer_count(texture: &wgpu::Texture, multiview_stereo: bool) -> u32 {
    let layers_in_texture = texture.size().depth_or_array_layers.max(1);
    if multiview_stereo {
        2.min(layers_in_texture)
    } else {
        1
    }
}

/// Creates a sampled `D2Array` view over one or two layers of `texture`.
pub(crate) fn create_d2_array_view(
    texture: &wgpu::Texture,
    label: &str,
    multiview_stereo: bool,
) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some(label),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        array_layer_count: Some(d2_array_layer_count(texture, multiview_stereo)),
        ..Default::default()
    })
}

/// Builds a standard fullscreen triangle render pipeline.
pub(crate) fn create_fullscreen_render_pipeline(
    device: &wgpu::Device,
    desc: FullscreenRenderPipelineDesc<'_>,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(desc.label),
        bind_group_layouts: desc.bind_group_layouts,
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(desc.label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: desc.shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: desc.shader,
            entry_point: Some(desc.fragment_entry),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: desc.output_format,
                blend: desc.blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: multiview_mask(desc.multiview_stereo),
        cache: None,
    })
}

/// Generic locked cache with double-check insertion and optional clear-on-overflow eviction.
#[derive(Debug)]
struct GpuCacheMap<K, V> {
    /// Cached objects keyed by pass-specific descriptors.
    entries: Mutex<HashMap<K, V>>,
    /// Maximum number of entries retained before the map is cleared.
    max_entries: Option<usize>,
}

impl<K, V> Default for GpuCacheMap<K, V> {
    fn default() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            max_entries: None,
        }
    }
}

impl<K, V> GpuCacheMap<K, V> {
    /// Creates an empty unbounded map.
    fn new() -> Self {
        Self::default()
    }

    /// Creates an empty map that clears itself before inserting once it reaches `max_entries`.
    fn with_max_entries(max_entries: usize) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            max_entries: Some(max_entries),
        }
    }
}

impl<K, V> GpuCacheMap<K, V>
where
    K: Clone + Eq + Hash,
    V: Clone,
{
    /// Returns a cached value or builds, double-checks, and inserts a new one.
    fn get_or_create(&self, key: K, build: impl FnOnce(&K) -> V) -> V {
        {
            let guard = self.entries.lock();
            if let Some(existing) = guard.get(&key) {
                return existing.clone();
            }
        }

        let value = build(&key);
        let mut guard = self.entries.lock();
        if let Some(existing) = guard.get(&key) {
            return existing.clone();
        }
        if self
            .max_entries
            .is_some_and(|max_entries| guard.len() >= max_entries)
        {
            guard.clear();
        }
        guard.insert(key, value.clone());
        value
    }

    /// Clears all cached entries.
    #[cfg(test)]
    fn clear(&self) {
        self.entries.lock().clear();
    }

    /// Returns the number of cached entries.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.lock().len()
    }
}

/// Typed cache for `wgpu::RenderPipeline` values.
#[derive(Debug)]
pub(crate) struct RenderPipelineMap<K> {
    /// Shared map storing pipelines behind `Arc` so record paths can clone handles cheaply.
    inner: GpuCacheMap<K, Arc<wgpu::RenderPipeline>>,
}

impl<K> Default for RenderPipelineMap<K> {
    fn default() -> Self {
        Self {
            inner: GpuCacheMap::new(),
        }
    }
}

impl<K> RenderPipelineMap<K>
where
    K: Clone + Eq + Hash,
{
    /// Returns a cached render pipeline or builds one outside the map lock.
    pub(crate) fn get_or_create(
        &self,
        key: K,
        build: impl FnOnce(&K) -> wgpu::RenderPipeline,
    ) -> Arc<wgpu::RenderPipeline> {
        self.inner.get_or_create(key, |key| Arc::new(build(key)))
    }
}

/// Typed cache for `wgpu::BindGroup` values.
#[derive(Debug)]
pub(crate) struct BindGroupMap<K> {
    /// Shared map storing bind groups keyed by pass-specific resource identity.
    inner: GpuCacheMap<K, wgpu::BindGroup>,
}

impl<K> Default for BindGroupMap<K> {
    fn default() -> Self {
        Self {
            inner: GpuCacheMap::new(),
        }
    }
}

impl<K> BindGroupMap<K>
where
    K: Clone + Eq + Hash,
{
    /// Creates an empty bind-group map with clear-on-overflow eviction.
    pub(crate) fn with_max_entries(max_entries: usize) -> Self {
        Self {
            inner: GpuCacheMap::with_max_entries(max_entries),
        }
    }

    /// Returns a cached bind group or builds one outside the map lock.
    pub(crate) fn get_or_create(
        &self,
        key: K,
        build: impl FnOnce(&K) -> wgpu::BindGroup,
    ) -> wgpu::BindGroup {
        self.inner.get_or_create(key, build)
    }
}

#[cfg(test)]
mod tests {
    use super::GpuCacheMap;

    #[test]
    fn cache_separates_keys_and_reuses_values() {
        let cache = GpuCacheMap::<u32, u32>::new();

        assert_eq!(cache.get_or_create(1, |_| 10), 10);
        assert_eq!(cache.get_or_create(1, |_| 20), 10);
        assert_eq!(cache.get_or_create(2, |_| 20), 20);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn bounded_cache_clears_before_overflow_insert() {
        let cache = GpuCacheMap::<u32, u32>::with_max_entries(2);

        assert_eq!(cache.get_or_create(1, |_| 10), 10);
        assert_eq!(cache.get_or_create(2, |_| 20), 20);
        assert_eq!(cache.len(), 2);

        assert_eq!(cache.get_or_create(3, |_| 30), 30);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get_or_create(1, |_| 40), 40);
    }

    #[test]
    fn cache_clear_drops_entries() {
        let cache = GpuCacheMap::<u32, u32>::new();

        cache.get_or_create(1, |_| 10);
        cache.clear();

        assert_eq!(cache.len(), 0);
    }
}
