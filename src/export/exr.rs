//! OpenEXR export (multi-channel, float).
//!
//! Writes one `.exr` file per cube face with named channels.
//! Channel presence is configurable; missing channels can optionally be written as 0.0 for schema stability.

use std::path::Path;

use exr::image::{
    AnyChannel, AnyChannels, FlatSamples, Image, Layer,
};
use exr::meta::header::LayerAttributes;
use exr::prelude::{Encoding, WritableImage};
use thiserror::Error;

use crate::terrain::{CubeFace, Planet};

/// Errors that can occur during EXR export.
#[derive(Error, Debug)]
pub enum ExrExportError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("IO / EXR error: {0}")]
    Exr(#[from] exr::error::Error),
    #[error("Invalid channel data length for '{name}': got {got}, expected {expected}")]
    InvalidChannelLength {
        name: &'static str,
        got: usize,
        expected: usize,
    },
}

/// Channel preset for EXR export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExrChannelsPreset {
    /// Only `height`.
    HeightOnly,
    /// Write `height` + all other channels that are currently available on the face.
    AllAvailable,
    /// Always write the full schema; missing channels are filled with 0.0.
    StableSchema,
}

impl Default for ExrChannelsPreset {
    fn default() -> Self {
        ExrChannelsPreset::AllAvailable
    }
}

/// Options for EXR export.
#[derive(Debug, Clone)]
pub struct ExrExportOptions {
    pub preset: ExrChannelsPreset,
    /// Compression/encoding choice.
    pub encoding: Encoding,
    /// Layer name in the EXR file.
    pub layer_name: &'static str,
}

impl Default for ExrExportOptions {
    fn default() -> Self {
        Self {
            preset: ExrChannelsPreset::AllAvailable,
            encoding: Encoding::FAST_LOSSLESS,
            layer_name: "worldgen",
        }
    }
}

fn expected_len(face: &CubeFace) -> usize {
    (face.resolution as usize) * (face.resolution as usize)
}

fn push_f32_channel(
    channels: &mut Vec<AnyChannel<FlatSamples>>,
    name: &'static str,
    samples: Vec<f32>,
    expected: usize,
) -> std::result::Result<(), ExrExportError> {
    if samples.len() != expected {
        return Err(ExrExportError::InvalidChannelLength {
            name,
            got: samples.len(),
            expected,
        });
    }
    channels.push(AnyChannel::new(name, FlatSamples::F32(samples)));
    Ok(())
}

fn channel_or_zeros_f32(
    opt: Option<&Vec<f32>>,
    expected: usize,
) -> Vec<f32> {
    match opt {
        Some(v) if v.len() == expected => v.clone(),
        _ => vec![0.0; expected],
    }
}

fn channel_or_zeros_u8_to_f32(
    opt: Option<&Vec<u8>>,
    expected: usize,
    scale: f32,
) -> Vec<f32> {
    match opt {
        Some(v) if v.len() == expected => v.iter().map(|&b| (b as f32) * scale).collect(),
        _ => vec![0.0; expected],
    }
}

fn channel_or_zeros_u32_to_f32(
    opt: Option<&Vec<u32>>,
    expected: usize,
) -> Vec<f32> {
    match opt {
        Some(v) if v.len() == expected => v.iter().map(|&x| x as f32).collect(),
        _ => vec![0.0; expected],
    }
}

fn channel_or_zeros_usize_to_f32(
    opt: Option<&Vec<usize>>,
    expected: usize,
) -> Vec<f32> {
    match opt {
        Some(v) if v.len() == expected => v.iter().map(|&x| x as f32).collect(),
        _ => vec![0.0; expected],
    }
}

fn build_channels_for_face(
    face: &CubeFace,
    options: &ExrExportOptions,
) -> std::result::Result<Vec<AnyChannel<FlatSamples>>, ExrExportError> {
    let n = expected_len(face);
    let mut channels: Vec<AnyChannel<FlatSamples>> = Vec::new();

    // Always present.
    push_f32_channel(&mut channels, "height", face.heights.clone(), n)?;

    if options.preset == ExrChannelsPreset::HeightOnly {
        return Ok(channels);
    }

    let stable = options.preset == ExrChannelsPreset::StableSchema;

    // Tectonics.
    if stable || face.tectonic_uplift.is_some() {
        let samples = if stable {
            channel_or_zeros_f32(face.tectonic_uplift.as_ref(), n)
        } else {
            face.tectonic_uplift.as_ref().unwrap().clone()
        };
        push_f32_channel(&mut channels, "tectonic_uplift", samples, n)?;
    }
    if stable || face.plate_ids.is_some() {
        let samples = if stable {
            channel_or_zeros_usize_to_f32(face.plate_ids.as_ref(), n)
        } else {
            face.plate_ids.as_ref().unwrap().iter().map(|&x| x as f32).collect()
        };
        push_f32_channel(&mut channels, "plate_id", samples, n)?;
    }

    // Erosion.
    if stable || face.water.is_some() {
        let samples = if stable {
            channel_or_zeros_f32(face.water.as_ref(), n)
        } else {
            face.water.as_ref().unwrap().clone()
        };
        push_f32_channel(&mut channels, "water", samples, n)?;
    }
    if stable || face.sediment.is_some() {
        let samples = if stable {
            channel_or_zeros_f32(face.sediment.as_ref(), n)
        } else {
            face.sediment.as_ref().unwrap().clone()
        };
        push_f32_channel(&mut channels, "sediment", samples, n)?;
    }
    if stable || face.deposition.is_some() {
        let samples = if stable {
            channel_or_zeros_f32(face.deposition.as_ref(), n)
        } else {
            face.deposition.as_ref().unwrap().clone()
        };
        push_f32_channel(&mut channels, "deposition", samples, n)?;
    }
    if stable || face.flow_accum.is_some() {
        let samples = if stable {
            channel_or_zeros_u32_to_f32(face.flow_accum.as_ref(), n)
        } else {
            face.flow_accum.as_ref().unwrap().iter().map(|&x| x as f32).collect()
        };
        push_f32_channel(&mut channels, "flow_accum", samples, n)?;
    }
    if stable || face.river_mask.is_some() {
        // Store as 0.0 / 1.0.
        let samples = if stable {
            channel_or_zeros_u8_to_f32(face.river_mask.as_ref(), n, 1.0 / 255.0)
        } else {
            face.river_mask
                .as_ref()
                .unwrap()
                .iter()
                .map(|&b| (b as f32) * (1.0 / 255.0))
                .collect()
        };
        push_f32_channel(&mut channels, "river_mask", samples, n)?;
    }

    // Climate (annual summaries).
    if stable || face.coast_distance_km.is_some() {
        let samples = if stable {
            channel_or_zeros_f32(face.coast_distance_km.as_ref(), n)
        } else {
            face.coast_distance_km.as_ref().unwrap().clone()
        };
        push_f32_channel(&mut channels, "coast_km", samples, n)?;
    }
    if stable || face.temperature_mean_c.is_some() {
        let samples = if stable {
            channel_or_zeros_f32(face.temperature_mean_c.as_ref(), n)
        } else {
            face.temperature_mean_c.as_ref().unwrap().clone()
        };
        push_f32_channel(&mut channels, "temp_mean_c", samples, n)?;
    }
    if stable || face.temp_min_month_c.is_some() {
        let samples = if stable {
            channel_or_zeros_f32(face.temp_min_month_c.as_ref(), n)
        } else {
            face.temp_min_month_c.as_ref().unwrap().clone()
        };
        push_f32_channel(&mut channels, "temp_min_c", samples, n)?;
    }
    if stable || face.temp_max_month_c.is_some() {
        let samples = if stable {
            channel_or_zeros_f32(face.temp_max_month_c.as_ref(), n)
        } else {
            face.temp_max_month_c.as_ref().unwrap().clone()
        };
        push_f32_channel(&mut channels, "temp_max_c", samples, n)?;
    }
    if stable || face.precip_annual_mm.is_some() {
        let samples = if stable {
            channel_or_zeros_f32(face.precip_annual_mm.as_ref(), n)
        } else {
            face.precip_annual_mm.as_ref().unwrap().clone()
        };
        push_f32_channel(&mut channels, "precip_annual_mm", samples, n)?;
    }

    // Biomes.
    if stable || face.land_mask.is_some() {
        // Store as 0.0 / 1.0.
        let samples = if stable {
            channel_or_zeros_u8_to_f32(face.land_mask.as_ref(), n, 1.0 / 255.0)
        } else {
            face.land_mask
                .as_ref()
                .unwrap()
                .iter()
                .map(|&b| (b as f32) * (1.0 / 255.0))
                .collect()
        };
        push_f32_channel(&mut channels, "land_mask", samples, n)?;
    }
    if stable || face.biome_ids.is_some() {
        let samples = if stable {
            channel_or_zeros_u8_to_f32(face.biome_ids.as_ref(), n, 1.0)
        } else {
            face.biome_ids
                .as_ref()
                .unwrap()
                .iter()
                .map(|&b| b as f32)
                .collect()
        };
        push_f32_channel(&mut channels, "biome_id", samples, n)?;
    }
    if stable || face.roughness.is_some() {
        let samples = if stable {
            channel_or_zeros_f32(face.roughness.as_ref(), n)
        } else {
            face.roughness.as_ref().unwrap().clone()
        };
        push_f32_channel(&mut channels, "roughness", samples, n)?;
    }
    if stable || face.albedo.is_some() {
        let samples = if stable {
            channel_or_zeros_f32(face.albedo.as_ref(), n)
        } else {
            face.albedo.as_ref().unwrap().clone()
        };
        push_f32_channel(&mut channels, "albedo", samples, n)?;
    }
    if stable || face.vegetation_density.is_some() {
        let samples = if stable {
            channel_or_zeros_f32(face.vegetation_density.as_ref(), n)
        } else {
            face.vegetation_density.as_ref().unwrap().clone()
        };
        push_f32_channel(&mut channels, "vegetation", samples, n)?;
    }

    Ok(channels)
}

/// Export a single cube face to OpenEXR with multi-channel float outputs.
pub fn export_face_exr(
    face: &CubeFace,
    path: &Path,
    options: &ExrExportOptions,
) -> std::result::Result<(), ExrExportError> {
    let n = expected_len(face);
    let channels = build_channels_for_face(face, options)?;

    // The exr crate expects channels in a consistent order for writing.
    // Some versions expose `AnyChannels::sort` for Vec, others for SmallVec; rely on Vec here
    // and adjust if the compiler requires SmallVec.
    let any_channels = AnyChannels::sort(channels.into());

    let layer = Layer::new(
        (face.resolution as usize, face.resolution as usize),
        LayerAttributes::named(options.layer_name),
        options.encoding,
        any_channels,
    );

    let image = Image::from_layer(layer);
    // Sanity: ensure file write matches expected pixel count.
    debug_assert_eq!(n, (face.resolution as usize) * (face.resolution as usize));
    image.write().to_file(path)?;
    Ok(())
}

/// Export all faces of a planet as individual OpenEXR files.
///
/// Files are named using the pattern: `{base_name}_{face_name}.exr`.
pub fn export_planet_exr(
    planet: &Planet,
    output_dir: &Path,
    base_name: &str,
    options: &ExrExportOptions,
) -> std::result::Result<(), ExrExportError> {
    std::fs::create_dir_all(output_dir)?;

    for face in &planet.faces {
        let filename = format!("{}_{}.exr", base_name, face.id.short_name());
        let path = output_dir.join(filename);
        export_face_exr(face, &path, options)?;
    }

    Ok(())
}

