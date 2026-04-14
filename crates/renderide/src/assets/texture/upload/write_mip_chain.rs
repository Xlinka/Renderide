//! Full mip chain path: decode, optional flip, [`super::mip_write_common::write_one_mip`] per level.

use crate::shared::{SetTexture2DData, SetTexture2DFormat};

use super::super::decode::{decode_mip_to_rgba8, flip_mip_rows, needs_rgba8_decode_before_upload};
use super::super::layout::{
    host_format_is_compressed, mip_byte_len, mip_dimensions_at_level, mip_tight_bytes_per_texel,
};
use super::mip_write_common::{
    choose_mip_start_bias, is_rgba8_family, uncompressed_row_bytes, write_one_mip,
};
use super::subregion::{hint_region_is_empty, try_write_texture2d_subregion};

/// Uploads mips from `raw` (exact shared-memory descriptor window) into `texture` using `wgpu_format`.
pub fn write_texture2d_mips(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    fmt: &SetTexture2DFormat,
    wgpu_format: wgpu::TextureFormat,
    upload: &SetTexture2DData,
    raw: &[u8],
) -> Result<u32, String> {
    if upload.hint.has_region != 0 {
        if hint_region_is_empty(&upload.hint) {
            logger::trace!(
                "texture {}: TextureUploadHint.has_region set but region empty; skipping upload",
                upload.asset_id
            );
            return Ok(0);
        }
        match try_write_texture2d_subregion(queue, texture, fmt, wgpu_format, upload, raw) {
            Some(Ok(n)) => {
                logger::trace!(
                    "texture {}: sub-region texture upload ({} mips equivalent)",
                    upload.asset_id,
                    n
                );
                return Ok(n);
            }
            Some(Err(e)) => return Err(e),
            None => {
                logger::trace!(
                    "texture {}: TextureUploadHint.has_region set; using full mip upload path",
                    upload.asset_id
                );
            }
        }
    }
    let want = upload.data.length.max(0) as usize;
    if raw.len() < want {
        return Err(format!(
            "raw shorter than descriptor (need {want}, got {})",
            raw.len()
        ));
    }
    let payload = &raw[..want];

    let start_base = upload.start_mip_level.max(0) as u32;
    let mipmap_count = fmt.mipmap_count.max(1) as u32;
    if start_base >= mipmap_count {
        return Err(format!(
            "start_mip_level {start_base} >= mipmap_count {mipmap_count}"
        ));
    }

    let flip = upload.flip_y;
    if flip && host_format_is_compressed(fmt.format) && !is_rgba8_family(wgpu_format) {
        logger::warn!(
            "texture {}: flip_y unsupported for compressed GPU texture {:?}; mips may look upside-down",
            upload.asset_id,
            wgpu_format
        );
    }

    let tex_extent = texture.size();
    let fmt_w = fmt.width.max(0) as u32;
    let fmt_h = fmt.height.max(0) as u32;
    if tex_extent.width != fmt_w || tex_extent.height != fmt_h {
        return Err(format!(
            "GPU texture {}x{} does not match SetTexture2DFormat {}x{} for asset {}",
            tex_extent.width, tex_extent.height, fmt_w, fmt_h, upload.asset_id
        ));
    }

    if upload.mip_map_sizes.len() != upload.mip_starts.len() {
        return Err("mip_map_sizes and mip_starts length mismatch".into());
    }
    if upload.mip_map_sizes.is_empty() {
        return Err("no mips in upload".into());
    }

    let (start_bias, valid_prefix_mips) = choose_mip_start_bias(fmt.format, upload, payload.len())?;
    if start_bias != 0 {
        logger::debug!(
            "texture {}: rebasing mip_starts by descriptor offset {}",
            upload.asset_id,
            start_bias
        );
    }

    let mut uploaded_mips = 0u32;
    for (i, sz) in upload.mip_map_sizes.iter().enumerate() {
        let w = sz.x.max(0) as u32;
        let h = sz.y.max(0) as u32;
        let mip_level = start_base + i as u32;
        if mip_level >= mipmap_count {
            return Err(format!(
                "upload mip {mip_level} exceeds texture mips {mipmap_count}"
            ));
        }

        let (gw, gh) = mip_dimensions_at_level(tex_extent.width, tex_extent.height, mip_level);
        if w != gw || h != gh {
            return Err(format!(
                "texture {} mip {mip_level}: upload says {w}x{h} but GPU mip is {gw}x{gh} (base {}x{} from format); fix host SetTexture2DFormat vs SetTexture2DData",
                upload.asset_id,
                tex_extent.width,
                tex_extent.height
            ));
        }

        let start_raw = upload.mip_starts[i];
        if start_raw < 0 {
            if uploaded_mips == 0 {
                return Err("negative mip_starts".into());
            }
            logger::warn!(
                "texture {}: uploaded {uploaded_mips}/{} mips; stopping at mip {} because mip_starts is negative",
                upload.asset_id,
                upload.mip_map_sizes.len(),
                i
            );
            break;
        }
        let start_abs = start_raw as usize;
        if start_abs < start_bias {
            if uploaded_mips == 0 {
                return Err(format!(
                    "mip 0 start {} is before descriptor offset {}",
                    start_abs, start_bias
                ));
            }
            logger::warn!(
                "texture {}: uploaded {uploaded_mips}/{} mips; stopping at mip {} because start {} is before descriptor offset {}",
                upload.asset_id,
                upload.mip_map_sizes.len(),
                i,
                start_abs,
                start_bias
            );
            break;
        }
        let start = start_abs - start_bias;
        let host_len = mip_byte_len(fmt.format, w, h)
            .ok_or_else(|| format!("mip byte size unsupported for {:?}", fmt.format))?
            as usize;
        let Some(mip_src) = payload.get(start..start + host_len) else {
            if uploaded_mips == 0 {
                return Err(format!(
                    "mip 0 slice out of range after rebasing by {start_bias} (payload_len={}, valid_prefix_mips={valid_prefix_mips})",
                    payload.len()
                ));
            }
            logger::warn!(
                "texture {}: uploaded {uploaded_mips}/{} mips; stopping at mip {} because payload_len={} does not cover start={} len={} after rebasing by {}",
                upload.asset_id,
                upload.mip_map_sizes.len(),
                i,
                payload.len(),
                start,
                host_len,
                start_bias
            );
            break;
        };

        let pixels: std::borrow::Cow<'_, [u8]> = if is_rgba8_family(wgpu_format) {
            if needs_rgba8_decode_before_upload(fmt.format) || host_format_is_compressed(fmt.format)
            {
                std::borrow::Cow::Owned(
                    decode_mip_to_rgba8(fmt.format, w, h, flip, mip_src).ok_or_else(|| {
                        format!("RGBA decode failed for mip {i} ({:?})", fmt.format)
                    })?,
                )
            } else if flip {
                let mut v = mip_src.to_vec();
                let bpp = mip_tight_bytes_per_texel(v.len(), w, h).ok_or_else(|| {
                    format!(
                        "mip {i}: RGBA8 upload len {} not divisible by {}×{} texels",
                        v.len(),
                        w,
                        h
                    )
                })?;
                if bpp != 4 {
                    return Err(format!(
                        "mip {i}: RGBA8 family expects 4 bytes per texel, got {bpp}"
                    ));
                }
                flip_mip_rows(&mut v, w, h, bpp);
                std::borrow::Cow::Owned(v)
            } else {
                std::borrow::Cow::Borrowed(mip_src)
            }
        } else {
            if needs_rgba8_decode_before_upload(fmt.format) {
                return Err(format!(
                    "host {:?} must use RGBA decode but GPU format is {:?}",
                    fmt.format, wgpu_format
                ));
            }
            if flip && !host_format_is_compressed(fmt.format) {
                let mut v = mip_src.to_vec();
                let bpp_host = mip_tight_bytes_per_texel(v.len(), w, h).ok_or_else(|| {
                    format!(
                        "mip {i}: len {} not divisible by {}×{} texels (cannot infer row stride for flip_y)",
                        v.len(),
                        w,
                        h
                    )
                })?;
                if let Ok(bpp_gpu) = uncompressed_row_bytes(wgpu_format) {
                    if bpp_host != bpp_gpu {
                        logger::warn!(
                            "texture {} mip {i}: host texel stride {} B != GPU {:?} stride {} B; flip_y uses host packing",
                            upload.asset_id,
                            bpp_host,
                            wgpu_format,
                            bpp_gpu
                        );
                    }
                }
                flip_mip_rows(&mut v, w, h, bpp_host);
                std::borrow::Cow::Owned(v)
            } else {
                if flip && host_format_is_compressed(fmt.format) {
                    logger::warn!(
                        "texture {} mip {i}: flip_y skipped for compressed {:?} GPU upload",
                        upload.asset_id,
                        wgpu_format
                    );
                }
                std::borrow::Cow::Borrowed(mip_src)
            }
        };

        write_one_mip(
            queue,
            texture,
            mip_level,
            w,
            h,
            wgpu_format,
            pixels.as_ref(),
        )?;
        uploaded_mips += 1;
    }

    if uploaded_mips == 0 {
        return Err("no mip levels uploaded".into());
    }
    Ok(uploaded_mips)
}
