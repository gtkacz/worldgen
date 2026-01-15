//! Equirectangular (lat/lon) PNG exports assembled from the cube-sphere data.
//!
//! This is useful for quick viewing in standard map tools, and as an intermediate
//! format for downstream pipelines.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use glam::Vec3;
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ImageBuffer, ImageEncoder, Luma, Rgb};
use thiserror::Error;

use crate::biomes::biome_preview_rgb;
use crate::geometry::sphere_to_face_uv;
use crate::terrain::Planet;

/// Errors that can occur during equirectangular export.
#[derive(Error, Debug)]
pub enum EquirectExportError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Image encoding error: {0}")]
    Image(#[from] image::ImageError),
    #[error("Invalid value range: min ({0}) >= max ({1})")]
    InvalidRange(f32, f32),
    #[error("Invalid output dimensions: {0}x{1}")]
    InvalidDimensions(u32, u32),
    #[error("No biome data available - run biome stage first")]
    NoBiomeData,
}

/// Options for equirectangular export.
#[derive(Debug, Clone)]
pub struct EquirectExportOptions {
    /// Output width in pixels. If None, defaults to `4 * planet.resolution()`.
    pub width: Option<u32>,
    /// Output height in pixels. If None, defaults to `2 * planet.resolution()`.
    pub height: Option<u32>,
    /// PNG compression type.
    pub compression: CompressionType,
    /// PNG filter type.
    pub filter: FilterType,
}

impl Default for EquirectExportOptions {
    fn default() -> Self {
        Self {
            width: None,
            height: None,
            compression: CompressionType::Default,
            filter: FilterType::Adaptive,
        }
    }
}

fn resolve_dims(planet: &Planet, options: &EquirectExportOptions) -> Result<(u32, u32), EquirectExportError> {
    let w = options.width.unwrap_or_else(|| planet.resolution().saturating_mul(4));
    let h = options.height.unwrap_or_else(|| planet.resolution().saturating_mul(2));
    if w < 2 || h < 2 {
        return Err(EquirectExportError::InvalidDimensions(w, h));
    }
    Ok((w, h))
}

#[inline]
fn wrap_lon(mut lon: f32) -> f32 {
    // Wrap to [-pi, pi]
    const TWO_PI: f32 = std::f32::consts::PI * 2.0;
    while lon > std::f32::consts::PI {
        lon -= TWO_PI;
    }
    while lon < -std::f32::consts::PI {
        lon += TWO_PI;
    }
    lon
}

#[inline]
fn lat_lon_to_dir(lat: f32, lon: f32) -> Vec3 {
    // lon=0 should point toward +Z (posz face center)
    let (slon, clon) = lon.sin_cos();
    let (slat, clat) = lat.sin_cos();
    Vec3::new(clat * slon, slat, clat * clon)
}

/// Export an equirectangular 16-bit PNG heightmap (normalized to `min_height..max_height`).
///
/// Output is written to `{output_dir}/{base_name}_equirect_height.png`.
pub fn export_planet_equirect_height_png(
    planet: &Planet,
    output_dir: &Path,
    base_name: &str,
    min_height: f32,
    max_height: f32,
    options: &EquirectExportOptions,
) -> Result<(), EquirectExportError> {
    if min_height >= max_height {
        return Err(EquirectExportError::InvalidRange(min_height, max_height));
    }

    std::fs::create_dir_all(output_dir)?;
    let (width, height) = resolve_dims(planet, options)?;

    let range = max_height - min_height;
    let mut img: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::new(width, height);

    for y in 0..height {
        // lat in [pi/2, -pi/2] (top to bottom), sampling pixel centers
        let fy = (y as f32 + 0.5) / height as f32;
        let lat = (std::f32::consts::FRAC_PI_2) - fy * std::f32::consts::PI;
        for x in 0..width {
            // lon in [-pi, pi] (left to right), sampling pixel centers
            let fx = (x as f32 + 0.5) / width as f32;
            let lon = wrap_lon(-std::f32::consts::PI + fx * (std::f32::consts::PI * 2.0));
            let dir = lat_lon_to_dir(lat, lon);

            let (face_id, u, v) = sphere_to_face_uv(dir);
            let face = planet.face(face_id);
            let (px, py) = face.uv_to_pixel(u, v);
            let h = face.get_height(px, py);

            let normalized = ((h - min_height) / range).clamp(0.0, 1.0);
            let value = (normalized * 65535.0) as u16;
            img.put_pixel(x, y, Luma([value]));
        }
    }

    let filename = format!("{}_equirect_height.png", base_name);
    let path = output_dir.join(filename);
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let encoder = PngEncoder::new_with_quality(writer, options.compression, options.filter);

    let raw_data = img.as_raw();
    let byte_slice: &[u8] = bytemuck::cast_slice(raw_data);
    encoder.write_image(
        byte_slice,
        width,
        height,
        image::ExtendedColorType::L16,
    )?;

    Ok(())
}

/// Export an equirectangular RGB biome preview PNG.
///
/// Output is written to `{output_dir}/{base_name}_equirect_biomes.png`.
pub fn export_planet_equirect_biomes_png(
    planet: &Planet,
    output_dir: &Path,
    base_name: &str,
    ocean_color: [u8; 3],
    options: &EquirectExportOptions,
) -> Result<(), EquirectExportError> {
    // Ensure biome data exists.
    if planet.faces.iter().any(|f| f.biome_ids.is_none()) {
        return Err(EquirectExportError::NoBiomeData);
    }

    std::fs::create_dir_all(output_dir)?;
    let (width, height) = resolve_dims(planet, options)?;

    let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(width, height);

    for y in 0..height {
        let fy = (y as f32 + 0.5) / height as f32;
        let lat = (std::f32::consts::FRAC_PI_2) - fy * std::f32::consts::PI;
        for x in 0..width {
            let fx = (x as f32 + 0.5) / width as f32;
            let lon = wrap_lon(-std::f32::consts::PI + fx * (std::f32::consts::PI * 2.0));
            let dir = lat_lon_to_dir(lat, lon);

            let (face_id, u, v) = sphere_to_face_uv(dir);
            let face = planet.face(face_id);
            let (px, py) = face.uv_to_pixel(u, v);
            let idx = (py * face.resolution + px) as usize;
            let id = face.biome_ids.as_ref().unwrap()[idx];

            let mut c = biome_preview_rgb(id);
            if id == 0 {
                c = ocean_color;
            }
            img.put_pixel(x, y, Rgb(c));
        }
    }

    let filename = format!("{}_equirect_biomes.png", base_name);
    let path = output_dir.join(filename);
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let encoder = PngEncoder::new_with_quality(writer, options.compression, options.filter);
    encoder.write_image(
        img.as_raw(),
        width,
        height,
        image::ExtendedColorType::Rgb8,
    )?;

    Ok(())
}

