//! Normal map generation from heightmaps (Sobel).

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use glam::Vec3;
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ImageBuffer, ImageEncoder, Rgb};
use thiserror::Error;

use crate::terrain::{CubeFace, Planet};

/// Errors that can occur during normal map export.
#[derive(Error, Debug)]
pub enum NormalMapError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Image encoding error: {0}")]
    Image(#[from] image::ImageError),
    #[error("Invalid normal strength: {0} (must be > 0)")]
    InvalidStrength(f32),
}

/// Options for normal map generation.
#[derive(Debug, Clone)]
pub struct NormalMapOptions {
    /// Scales the effect of height gradients. Higher = stronger normals.
    pub strength: f32,
    pub compression: CompressionType,
    pub filter: FilterType,
}

impl Default for NormalMapOptions {
    fn default() -> Self {
        Self {
            strength: 2.0,
            compression: CompressionType::Default,
            filter: FilterType::Adaptive,
        }
    }
}

fn clamp_i32(v: i32, lo: i32, hi: i32) -> i32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

fn height_at_clamped(face: &CubeFace, x: i32, y: i32) -> f32 {
    let res = face.resolution as i32;
    let xi = clamp_i32(x, 0, res - 1) as u32;
    let yi = clamp_i32(y, 0, res - 1) as u32;
    face.get_height(xi, yi)
}

fn normal_from_sobel(face: &CubeFace, x: i32, y: i32, strength: f32) -> Vec3 {
    // Sobel kernels:
    // Gx =
    // [ -1  0  1 ]
    // [ -2  0  2 ]
    // [ -1  0  1 ]
    //
    // Gy =
    // [ -1 -2 -1 ]
    // [  0  0  0 ]
    // [  1  2  1 ]

    let tl = height_at_clamped(face, x - 1, y - 1);
    let tc = height_at_clamped(face, x, y - 1);
    let tr = height_at_clamped(face, x + 1, y - 1);
    let ml = height_at_clamped(face, x - 1, y);
    let mr = height_at_clamped(face, x + 1, y);
    let bl = height_at_clamped(face, x - 1, y + 1);
    let bc = height_at_clamped(face, x, y + 1);
    let br = height_at_clamped(face, x + 1, y + 1);

    let gx = (-1.0 * tl) + (1.0 * tr)
        + (-2.0 * ml) + (2.0 * mr)
        + (-1.0 * bl) + (1.0 * br);

    let gy = (-1.0 * tl) + (-2.0 * tc) + (-1.0 * tr)
        + (1.0 * bl) + (2.0 * bc) + (1.0 * br);

    // Convert gradient into a normal. Z is constant; scale gradients by strength.
    Vec3::new(-gx * strength, -gy * strength, 1.0).normalize_or_zero()
}

fn encode_normal_rgb8(n: Vec3) -> [u8; 3] {
    let c = (n * 0.5) + Vec3::splat(0.5);
    [
        (c.x.clamp(0.0, 1.0) * 255.0) as u8,
        (c.y.clamp(0.0, 1.0) * 255.0) as u8,
        (c.z.clamp(0.0, 1.0) * 255.0) as u8,
    ]
}

/// Export a single cube face normal map as RGB PNG.
///
/// Output is tangent-space-like with Z pointing “out of the image”.
pub fn export_face_normal_map_png(
    face: &CubeFace,
    path: &Path,
    options: &NormalMapOptions,
) -> Result<(), NormalMapError> {
    if !options.strength.is_finite() || options.strength <= 0.0 {
        return Err(NormalMapError::InvalidStrength(options.strength));
    }

    let res = face.resolution;
    let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(res, res);

    for y in 0..res {
        for x in 0..res {
            let n = normal_from_sobel(face, x as i32, y as i32, options.strength);
            img.put_pixel(x, y, Rgb(encode_normal_rgb8(n)));
        }
    }

    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let encoder = PngEncoder::new_with_quality(writer, options.compression, options.filter);
    encoder.write_image(
        img.as_raw(),
        res,
        res,
        image::ExtendedColorType::Rgb8,
    )?;
    Ok(())
}

/// Export all faces of a planet as normal map PNGs.
///
/// Files are named: `{base_name}_normal_{face}.png`.
pub fn export_planet_normal_maps_png(
    planet: &Planet,
    output_dir: &Path,
    base_name: &str,
    options: &NormalMapOptions,
) -> Result<(), NormalMapError> {
    std::fs::create_dir_all(output_dir)?;
    for face in &planet.faces {
        let filename = format!("{}_normal_{}.png", base_name, face.id.short_name());
        let path = output_dir.join(filename);
        export_face_normal_map_png(face, &path, options)?;
    }
    Ok(())
}

