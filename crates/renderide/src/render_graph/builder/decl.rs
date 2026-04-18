//! Internal types backing [`super::GraphBuilder`] declarations.

use super::super::ids::{GroupId, PassId};
use super::super::pass::{GroupScope, PassSetup, RenderPass};
use super::super::resources::{
    TransientArrayLayers, TransientExtent, TransientSampleCount, TransientTextureFormat,
};

pub(super) struct PassEntry {
    pub(super) group: GroupId,
    pub(super) pass: Box<dyn RenderPass>,
}

#[derive(Clone, Debug)]
pub(super) struct GroupEntry {
    pub(super) name: &'static str,
    pub(super) scope: GroupScope,
    pub(super) after: Vec<GroupId>,
}

pub(super) struct SetupEntry {
    pub(super) id: PassId,
    pub(super) group: GroupId,
    pub(super) name: String,
    pub(super) setup: PassSetup,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct TextureAliasKey {
    pub(super) format: TransientTextureFormat,
    pub(super) extent: TransientExtent,
    pub(super) mip_levels: u32,
    pub(super) sample_count: TransientSampleCount,
    pub(super) dimension: wgpu::TextureDimension,
    pub(super) array_layers: TransientArrayLayers,
    pub(super) usage_bits: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct BufferAliasKey {
    pub(super) size_policy: super::super::resources::BufferSizePolicy,
    pub(super) usage_bits: u64,
}
