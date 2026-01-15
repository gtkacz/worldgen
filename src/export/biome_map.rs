//! Biome preview map export (Phase 5).

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ImageBuffer, ImageEncoder, Rgb};
use thiserror::Error;

use crate::terrain::{CubeFace, Planet};
use crate::biomes::biome_preview_rgb;

/// Errors that can occur during biome map export.
#[derive(Error, Debug)]
pub enum BiomeMapError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Image encoding error: {0}")]
    Image(#[from] image::ImageError),
    #[error("No biome data available - run biome stage first")]
    NoBiomeData,
}

/// Options for biome map export.
#[derive(Debug, Clone)]
pub struct BiomeMapOptions {
    pub compression: CompressionType,
    pub filter: FilterType,
    /// RGB color used for oceans/non-land (biome_id==0).
    pub ocean_color: [u8; 3],
}

impl Default for BiomeMapOptions {
    fn default() -> Self {
        Self {
            compression: CompressionType::Default,
            filter: FilterType::Adaptive,
            ocean_color: [15, 40, 90],
        }
    }
}

/// Export a single face biome preview map as RGB PNG.
pub fn export_face_biome_map_png(
    face: &CubeFace,
    path: &Path,
    options: &BiomeMapOptions,
) -> Result<(), BiomeMapError> {
    let biome_ids = face
        .biome_ids
        .as_ref()
        .ok_or(BiomeMapError::NoBiomeData)?;

    let resolution = face.resolution;
    let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(resolution, resolution);

    for y in 0..resolution {
        for x in 0..resolution {
            let id = biome_ids[(y * resolution + x) as usize];
            let mut c = biome_preview_rgb(id);
            if id == 0 {
                c = options.ocean_color;
            }
            img.put_pixel(x, y, Rgb(c));
        }
    }

    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let encoder = PngEncoder::new_with_quality(writer, options.compression, options.filter);
    encoder.write_image(
        img.as_raw(),
        resolution,
        resolution,
        image::ExtendedColorType::Rgb8,
    )?;

    Ok(())
}

/// Export all faces of a planet as biome preview PNGs.
pub fn export_planet_biome_map_png(
    planet: &Planet,
    output_dir: &Path,
    base_name: &str,
    options: &BiomeMapOptions,
) -> Result<(), BiomeMapError> {
    std::fs::create_dir_all(output_dir)?;

    for face in &planet.faces {
        let filename = format!("{}_{}.png", base_name, face.id.short_name());
        let path = output_dir.join(filename);
        export_face_biome_map_png(face, &path, options)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::CubeFaceId;
    use tempfile::tempdir;

    #[test]
    fn export_biome_map_smoke() {
        let mut face = CubeFace::new(CubeFaceId::PosX, 16);
        let size = (16 * 16) as usize;
        face.biome_ids = Some(vec![4u8; size]);

        let dir = tempdir().unwrap();
        let path = dir.path().join("biomes.png");
        export_face_biome_map_png(&face, &path, &BiomeMapOptions::default()).unwrap();
        assert!(path.exists());
    }
}

