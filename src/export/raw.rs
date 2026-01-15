//! RAW format export for game engine compatibility.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use thiserror::Error;

use crate::terrain::{CubeFace, Planet};

/// Errors that can occur during RAW export.
#[derive(Error, Debug)]
pub enum RawExportError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid height range: min ({0}) >= max ({1})")]
    InvalidHeightRange(f32, f32),
}

/// RAW export format options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawFormat {
    /// 16-bit unsigned integer, little-endian (Unity default).
    R16LittleEndian,
    /// 16-bit unsigned integer, big-endian.
    R16BigEndian,
    /// 32-bit float, little-endian (high precision).
    R32Float,
}

impl Default for RawFormat {
    fn default() -> Self {
        RawFormat::R16LittleEndian
    }
}

/// Options for RAW export.
#[derive(Debug, Clone)]
pub struct RawExportOptions {
    /// RAW format to use.
    pub format: RawFormat,
    /// Minimum height for normalization (used for R16 formats).
    pub min_height: f32,
    /// Maximum height for normalization (used for R16 formats).
    pub max_height: f32,
}

impl Default for RawExportOptions {
    fn default() -> Self {
        Self {
            format: RawFormat::R16LittleEndian,
            min_height: -1.0,
            max_height: 1.0,
        }
    }
}

/// Exports a single cube face as a RAW heightmap.
///
/// # Arguments
/// * `face` - The cube face to export
/// * `path` - Output file path
/// * `format` - RAW format (R16 or R32)
/// * `min_height` - Minimum height for normalization (R16 only)
/// * `max_height` - Maximum height for normalization (R16 only)
///
/// # Returns
/// `Ok(())` on success, or an error if export fails
pub fn export_face_raw(
    face: &CubeFace,
    path: &Path,
    format: RawFormat,
    min_height: f32,
    max_height: f32,
) -> Result<(), RawExportError> {
    if format != RawFormat::R32Float && min_height >= max_height {
        return Err(RawExportError::InvalidHeightRange(min_height, max_height));
    }

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let range = max_height - min_height;

    match format {
        RawFormat::R16LittleEndian => {
            for &height in &face.heights {
                let normalized = ((height - min_height) / range).clamp(0.0, 1.0);
                let value = (normalized * 65535.0) as u16;
                writer.write_all(&value.to_le_bytes())?;
            }
        }
        RawFormat::R16BigEndian => {
            for &height in &face.heights {
                let normalized = ((height - min_height) / range).clamp(0.0, 1.0);
                let value = (normalized * 65535.0) as u16;
                writer.write_all(&value.to_be_bytes())?;
            }
        }
        RawFormat::R32Float => {
            for &height in &face.heights {
                writer.write_all(&height.to_le_bytes())?;
            }
        }
    }

    writer.flush()?;
    Ok(())
}

/// Exports all faces of a planet as individual RAW files.
///
/// Files are named using the pattern: `{base_name}_{face_name}.raw`
///
/// # Arguments
/// * `planet` - The planet to export
/// * `output_dir` - Directory to save files to
/// * `base_name` - Base name for output files
/// * `format` - RAW format to use
/// * `min_height` - Minimum height for normalization
/// * `max_height` - Maximum height for normalization
///
/// # Returns
/// `Ok(())` on success, or the first error encountered
pub fn export_planet_raw(
    planet: &Planet,
    output_dir: &Path,
    base_name: &str,
    format: RawFormat,
    min_height: f32,
    max_height: f32,
) -> Result<(), RawExportError> {
    std::fs::create_dir_all(output_dir)?;

    for face in &planet.faces {
        let filename = format!("{}_{}.raw", base_name, face.id.short_name());
        let path = output_dir.join(filename);
        export_face_raw(face, &path, format, min_height, max_height)?;
    }

    Ok(())
}

/// Returns the expected file size for a RAW export.
pub fn expected_file_size(resolution: u32, format: RawFormat) -> u64 {
    let pixels = (resolution as u64) * (resolution as u64);
    match format {
        RawFormat::R16LittleEndian | RawFormat::R16BigEndian => pixels * 2,
        RawFormat::R32Float => pixels * 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::CubeFaceId;
    use tempfile::tempdir;

    #[test]
    fn test_export_face_raw_r16() {
        let mut face = CubeFace::new(CubeFaceId::PosX, 64);
        for i in 0..face.heights.len() {
            face.heights[i] = (i as f32 / face.heights.len() as f32) * 2.0 - 1.0;
        }

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.raw");

        export_face_raw(&face, &path, RawFormat::R16LittleEndian, -1.0, 1.0).unwrap();

        assert!(path.exists());
        let metadata = std::fs::metadata(&path).unwrap();
        assert_eq!(metadata.len(), expected_file_size(64, RawFormat::R16LittleEndian));
    }

    #[test]
    fn test_export_face_raw_r32() {
        let mut face = CubeFace::new(CubeFaceId::NegY, 32);
        for i in 0..face.heights.len() {
            face.heights[i] = (i as f32 / face.heights.len() as f32) * 2.0 - 1.0;
        }

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.raw");

        export_face_raw(&face, &path, RawFormat::R32Float, -1.0, 1.0).unwrap();

        assert!(path.exists());
        let metadata = std::fs::metadata(&path).unwrap();
        assert_eq!(metadata.len(), expected_file_size(32, RawFormat::R32Float));
    }

    #[test]
    fn test_export_planet_raw() {
        let planet = Planet::new(16, 42, 6371.0);
        let dir = tempdir().unwrap();

        export_planet_raw(
            &planet,
            dir.path(),
            "terrain",
            RawFormat::R16LittleEndian,
            -1.0,
            1.0,
        ).unwrap();

        // Check all 6 face files were created
        for face_id in CubeFaceId::all() {
            let filename = format!("terrain_{}.raw", face_id.short_name());
            let path = dir.path().join(filename);
            assert!(path.exists(), "Missing file for {:?}", face_id);
        }
    }

    #[test]
    fn test_expected_file_size() {
        assert_eq!(expected_file_size(256, RawFormat::R16LittleEndian), 256 * 256 * 2);
        assert_eq!(expected_file_size(256, RawFormat::R32Float), 256 * 256 * 4);
    }

    #[test]
    fn test_raw_content_correctness() {
        let mut face = CubeFace::new(CubeFaceId::PosZ, 2);
        face.set_height(0, 0, -1.0);
        face.set_height(1, 0, 0.0);
        face.set_height(0, 1, 0.5);
        face.set_height(1, 1, 1.0);

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.raw");

        export_face_raw(&face, &path, RawFormat::R16LittleEndian, -1.0, 1.0).unwrap();

        // Read back and verify
        let data = std::fs::read(&path).unwrap();
        assert_eq!(data.len(), 8); // 4 pixels × 2 bytes

        // First pixel: -1.0 → 0
        let val0 = u16::from_le_bytes([data[0], data[1]]);
        assert_eq!(val0, 0);

        // Second pixel: 0.0 → 32767/32768
        let val1 = u16::from_le_bytes([data[2], data[3]]);
        assert!((val1 as i32 - 32767).abs() < 2);

        // Fourth pixel: 1.0 → 65535
        let val3 = u16::from_le_bytes([data[6], data[7]]);
        assert_eq!(val3, 65535);
    }
}
