//! Explicit group edges and resource read/write dependency edges.

use std::collections::{HashMap, HashSet};

use super::super::error::GraphBuildError;
use super::super::ids::PassId;
use super::super::pass::GroupScope;
use super::super::resources::{BufferResourceHandle, ResourceHandle, TextureResourceHandle};
use super::decl::SetupEntry;
use super::GraphBuilder;

pub(super) fn explicit_edges(
    builder: &GraphBuilder,
    n: usize,
) -> Result<HashSet<(usize, usize)>, GraphBuildError> {
    let mut edges = HashSet::new();
    for &(from, to) in &builder.edges {
        if from >= n || to >= n {
            return Err(GraphBuildError::InvalidEdge {
                from: PassId(from),
                to: PassId(to),
            });
        }
        if from != to {
            edges.insert((from, to));
        }
    }
    Ok(edges)
}

pub(super) fn add_group_edges(
    builder: &GraphBuilder,
    setups: &[SetupEntry],
    edges: &mut HashSet<(usize, usize)>,
) -> Result<(), GraphBuildError> {
    for entry in &builder.groups {
        for &dep in &entry.after {
            if dep.0 >= builder.groups.len() {
                return Err(GraphBuildError::CycleDetected);
            }
        }
    }

    for (a_idx, a) in setups.iter().enumerate() {
        for (b_idx, b) in setups.iter().enumerate() {
            if a_idx == b_idx {
                continue;
            }
            let a_group = &builder.groups[a.group.0];
            let b_group = &builder.groups[b.group.0];
            if a_group.scope == GroupScope::FrameGlobal && b_group.scope == GroupScope::PerView {
                edges.insert((a_idx, b_idx));
            }
            if b_group.after.contains(&a.group) {
                edges.insert((a_idx, b_idx));
            }
        }
    }
    Ok(())
}

pub(super) fn add_resource_edges(
    builder: &GraphBuilder,
    setups: &[SetupEntry],
    edges: &mut HashSet<(usize, usize)>,
) -> Result<(), GraphBuildError> {
    let mut by_resource: HashMap<ResourceHandle, Vec<(usize, bool, bool)>> = HashMap::new();
    for (pass_idx, setup) in setups.iter().enumerate() {
        for access in &setup.setup.accesses {
            by_resource.entry(access.resource).or_default().push((
                pass_idx,
                access.reads(),
                access.writes(),
            ));
        }
    }

    for (resource, mut accesses) in by_resource {
        accesses.sort_by_key(|(pass_idx, _, _)| *pass_idx);
        accesses.dedup();
        let mut last_writer: Option<usize> = None;
        let mut readers: HashSet<usize> = HashSet::new();
        for (pass_idx, reads, writes) in accesses {
            if reads {
                if let Some(writer) = last_writer {
                    if writer != pass_idx {
                        edges.insert((writer, pass_idx));
                    }
                } else if !resource.is_imported() {
                    return Err(GraphBuildError::MissingDependency {
                        pass: PassId(pass_idx),
                        resource: resource_label(builder, resource),
                    });
                }
                readers.insert(pass_idx);
            }
            if writes {
                if let Some(writer) = last_writer {
                    if writer != pass_idx {
                        edges.insert((writer, pass_idx));
                    }
                }
                for reader in readers.drain() {
                    if reader != pass_idx {
                        edges.insert((reader, pass_idx));
                    }
                }
                last_writer = Some(pass_idx);
            }
        }
    }
    Ok(())
}

fn resource_label(builder: &GraphBuilder, resource: ResourceHandle) -> String {
    match resource {
        ResourceHandle::Texture(TextureResourceHandle::Transient(h)) => builder
            .textures
            .get(h.index())
            .map(|d| d.label.to_string())
            .unwrap_or_else(|| format!("texture#{}", h.index())),
        ResourceHandle::Texture(TextureResourceHandle::Imported(h)) => builder
            .imports_tex
            .get(h.index())
            .map(|d| d.label.to_string())
            .unwrap_or_else(|| format!("imported_texture#{}", h.index())),
        ResourceHandle::Buffer(BufferResourceHandle::Transient(h)) => builder
            .buffers
            .get(h.index())
            .map(|d| d.label.to_string())
            .unwrap_or_else(|| format!("buffer#{}", h.index())),
        ResourceHandle::Buffer(BufferResourceHandle::Imported(h)) => builder
            .imports_buf
            .get(h.index())
            .map(|d| d.label.to_string())
            .unwrap_or_else(|| format!("imported_buffer#{}", h.index())),
    }
}
