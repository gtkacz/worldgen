//! PNG export functionality for heightmaps.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ImageBuffer, ImageEncoder, Luma};
use thiserror::Error;

use crate::terrain::{CubeFace, Planet};

/// Errors that can occur during PNG export.
#[derive(Error, Debug)]
pub enum PngExportError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Image encoding error: {0}")]
    Image(#[from] image::ImageError),
    #[error("Invalid height range: min ({0}) >= max ({1})")]
    InvalidHeightRange(f32, f32),
}

/// Options for PNG export.
#[derive(Debug, Clone)]
pub struct PngExportOptions {
    /// Minimum height value for normalization.
    pub min_height: f32,
    /// Maximum height value for normalization.
    pub max_height: f32,
    /// PNG compression type.
    pub compression: CompressionType,
    /// PNG filter type.
    pub filter: FilterType,
}

impl Default for PngExportOptions {
    fn default() -> Self {
        Self {
            min_height: -1.0,
            max_height: 1.0,
            compression: CompressionType::Default,
            filter: FilterType::Adaptive,
        }
    }
}

impl PngExportOptions {
    /// Creates options with auto-detected height range from the face.
    pub fn auto_range(face: &CubeFace) -> Self {
        let (min, max) = face.height_range();
        Self {
            min_height: min,
            max_height: max,
            ..Default::default()
        }
    }

    /// Creates options with auto-detected height range from the planet.
    pub fn auto_range_planet(planet: &Planet) -> Self {
        let (min, max) = planet.height_range();
        Self {
            min_height: min,
            max_height: max,
            ..Default::default()
        }
    }
}

/// Exports a single cube face as a 16-bit PNG heightmap.
///
/// # Arguments
/// * `face` - The cube face to export
/// * `path` - Output file path
/// * `options` - Export options including height range for normalization
///
/// # Returns
/// `Ok(())` on success, or an error if export fails
pub fn export_face_png(
    face: &CubeFace,
    path: &Path,
    options: &PngExportOptions,
) -> Result<(), PngExportError> {
    let min = options.min_height;
    let max = options.max_height;

    if min >= max {
        return Err(PngExportError::InvalidHeightRange(min, max));
    }

    let resolution = face.resolution;
    let range = max - min;

    // Create 16-bit grayscale image
    let mut img: ImageBuffer<Luma<u16>, Vec<u16>> =
        ImageBuffer::new(resolution, resolution);

    for y in 0..resolution {
        for x in 0..resolution {
            let height = face.get_height(x, y);
            // Normalize to [0, 1] then scale to u16
            let normalized = ((height - min) / range).clamp(0.0, 1.0);
            let value = (normalized * 65535.0) as u16;
            img.put_pixel(x, y, Luma([value]));
        }
    }

    // Write with specified compression settings
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let encoder = PngEncoder::new_with_quality(writer, options.compression, options.filter);

    // Convert u16 slice to bytes for the encoder
    let raw_data = img.as_raw();
    let byte_slice: &[u8] = bytemuck::cast_slice(raw_data);

    encoder.write_image(
        byte_slice,
        resolution,
        resolution,
        image::ExtendedColorType::L16,
    )?;

    Ok(())
}

/// Exports all faces of a planet as individual PNG files.
///
/// Files are named using the pattern: `{base_name}_{face_name}.png`
/// For example: `planet_posx.png`, `planet_negy.png`, etc.
///
/// # Arguments
/// * `planet` - The planet to export
/// * `output_dir` - Directory to save files to
/// * `base_name` - Base name for output files
/// * `options` - Export options
///
/// # Returns
/// `Ok(())` on success, or the first error encountered
pub fn export_planet_png(
    planet: &Planet,
    output_dir: &Path,
    base_name: &str,
    options: &PngExportOptions,
) -> Result<(), PngExportError> {
    std::fs::create_dir_all(output_dir)?;

    for face in &planet.faces {
        let filename = format!("{}_{}.png", base_name, face.id.short_name());
        let path = output_dir.join(filename);
        export_face_png(face, &path, options)?;
    }

    Ok(())
}

/// Export an arbitrary scalar field (f32) as a 16-bit grayscale PNG.
///
/// `data` must be length `resolution*resolution` in row-major order.
pub fn export_face_scalar_png_f32(
    resolution: u32,
    data: &[f32],
    path: &Path,
    min_value: f32,
    max_value: f32,
    compression: CompressionType,
    filter: FilterType,
) -> Result<(), PngExportError> {
    if min_value >= max_value {
        return Err(PngExportError::InvalidHeightRange(min_value, max_value));
    }
    let expected = (resolution as usize) * (resolution as usize);
    if data.len() != expected {
        return Err(PngExportError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("scalar data length {} != expected {}", data.len(), expected),
        )));
    }

    let range = max_value - min_value;
    let mut img: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::new(resolution, resolution);
    for y in 0..resolution {
        for x in 0..resolution {
            let v = data[(y * resolution + x) as usize];
            let normalized = ((v - min_value) / range).clamp(0.0, 1.0);
            let value = (normalized * 65535.0) as u16;
            img.put_pixel(x, y, Luma([value]));
        }
    }

    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let encoder = PngEncoder::new_with_quality(writer, compression, filter);
    let raw_data = img.as_raw();
    let byte_slice: &[u8] = bytemuck::cast_slice(raw_data);
    encoder.write_image(
        byte_slice,
        resolution,
        resolution,
        image::ExtendedColorType::L16,
    )?;
    Ok(())
}

/// Export a binary/byte mask (u8) as an 8-bit grayscale PNG.
///
/// `data` must be length `resolution*resolution` in row-major order.
pub fn export_face_mask_png_u8(
    resolution: u32,
    data: &[u8],
    path: &Path,
    compression: CompressionType,
    filter: FilterType,
) -> Result<(), PngExportError> {
    let expected = (resolution as usize) * (resolution as usize);
    if data.len() != expected {
        return Err(PngExportError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("mask data length {} != expected {}", data.len(), expected),
        )));
    }

    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let encoder = PngEncoder::new_with_quality(writer, compression, filter);
    encoder.write_image(
        data,
        resolution,
        resolution,
        image::ExtendedColorType::L8,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::CubeFaceId;
    use tempfile::tempdir;

    #[test]
    fn test_export_face_png() {
        let mut face = CubeFace::new(CubeFaceId::PosX, 64);
        // Create gradient for testing
        for y in 0..64 {
            for x in 0..64 {
                let height = (x as f32 + y as f32) / 126.0 * 2.0 - 1.0;
                face.set_height(x, y, height);
            }
        }

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.png");

        let options = PngExportOptions::default();
        export_face_png(&face, &path, &options).unwrap();

        assert!(path.exists());

        // Verify file size is reasonable
        let metadata = std::fs::metadata(&path).unwrap();
        assert!(metadata.len() > 0);
    }

    #[test]
    fn test_export_planet_png() {
        let planet = Planet::new(32, 42, 6371.0);
        let dir = tempdir().unwrap();

        let options = PngExportOptions::default();
        export_planet_png(&planet, dir.path(), "planet", &options).unwrap();

        // Check all 6 face files were created
        for face_id in CubeFaceId::all() {
            let filename = format!("planet_{}.png", face_id.short_name());
            let path = dir.path().join(filename);
            assert!(path.exists(), "Missing file for {:?}", face_id);
        }
    }

    #[test]
    fn test_invalid_height_range() {
        let face = CubeFace::new(CubeFaceId::PosZ, 16);
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.png");

        let options = PngExportOptions {
            min_height: 1.0,
            max_height: -1.0, // Invalid: min > max
            ..Default::default()
        };

        let result = export_face_png(&face, &path, &options);
        assert!(result.is_err());
    }

    #[test]
    fn test_auto_range() {
        let mut face = CubeFace::new(CubeFaceId::PosY, 16);
        face.set_height(0, 0, -0.5);
        face.set_height(15, 15, 0.75);

        let options = PngExportOptions::auto_range(&face);
        assert_eq!(options.min_height, -0.5);
        assert_eq!(options.max_height, 0.75);
    }
}
