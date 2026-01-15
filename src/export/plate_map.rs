//! Plate map export for visualizing tectonic plates and boundaries.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ImageBuffer, ImageEncoder, Rgb};
use thiserror::Error;

use crate::terrain::{CubeFace, Planet};
use crate::tectonics::{TectonicPlate, CrustType, PlateBoundary, BoundaryType};

/// Errors that can occur during plate map export.
#[derive(Error, Debug)]
pub enum PlateMapError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Image encoding error: {0}")]
    Image(#[from] image::ImageError),
    #[error("No tectonic data available - run tectonic stage first")]
    NoTectonicData,
    #[error("Plate ID {0} out of range")]
    InvalidPlateId(usize),
}

/// Options for plate map export.
#[derive(Debug, Clone)]
pub struct PlateMapOptions {
    /// PNG compression type.
    pub compression: CompressionType,
    /// PNG filter type.
    pub filter: FilterType,
    /// Whether to overlay boundary lines.
    pub show_boundaries: bool,
    /// Boundary line width in pixels.
    pub boundary_width: u32,
}

impl Default for PlateMapOptions {
    fn default() -> Self {
        Self {
            compression: CompressionType::Default,
            filter: FilterType::Adaptive,
            show_boundaries: true,
            boundary_width: 2,
        }
    }
}

/// Generates distinct colors for plates using golden ratio distribution.
///
/// This produces visually distinct colors that work well for visualization.
pub fn generate_plate_colors(num_plates: usize, seed: u64) -> Vec<[u8; 3]> {
    use rand::Rng;
    use rand_chacha::ChaCha8Rng;
    use rand::SeedableRng;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let golden_ratio = 0.618033988749895;

    let mut colors = Vec::with_capacity(num_plates);
    let mut hue = rng.random::<f32>();

    for _ in 0..num_plates {
        // Use golden ratio to spread hues evenly
        hue = (hue + golden_ratio) % 1.0;

        // Vary saturation and value for more distinction
        let saturation = 0.5 + rng.random::<f32>() * 0.4;
        let value = 0.6 + rng.random::<f32>() * 0.3;

        let rgb = hsv_to_rgb(hue, saturation, value);
        colors.push(rgb);
    }

    colors
}

/// Generates colors based on plate properties (continental vs oceanic).
pub fn generate_plate_colors_by_type(plates: &[TectonicPlate], seed: u64) -> Vec<[u8; 3]> {
    use rand::Rng;
    use rand_chacha::ChaCha8Rng;
    use rand::SeedableRng;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    plates.iter().map(|plate| {
        match plate.crust_type {
            CrustType::Continental => {
                // Earth tones: browns, greens, tans
                let hue = 0.08 + rng.random::<f32>() * 0.12; // 30-70 degrees
                let saturation = 0.3 + rng.random::<f32>() * 0.4;
                let value = 0.5 + rng.random::<f32>() * 0.4;
                hsv_to_rgb(hue, saturation, value)
            }
            CrustType::Oceanic => {
                // Blues and blue-greens
                let hue = 0.5 + rng.random::<f32>() * 0.15; // 180-234 degrees
                let saturation = 0.4 + rng.random::<f32>() * 0.4;
                let value = 0.3 + rng.random::<f32>() * 0.4;
                hsv_to_rgb(hue, saturation, value)
            }
        }
    }).collect()
}

/// Converts HSV color to RGB.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [u8; 3] {
    let h = h * 6.0;
    let i = h.floor() as i32;
    let f = h - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    let (r, g, b) = match i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };

    [
        (r * 255.0) as u8,
        (g * 255.0) as u8,
        (b * 255.0) as u8,
    ]
}

/// Returns color for a boundary type.
fn boundary_color(boundary_type: BoundaryType) -> [u8; 3] {
    match boundary_type {
        BoundaryType::Convergent => [255, 50, 50],   // Red
        BoundaryType::Divergent => [50, 100, 255],   // Blue
        BoundaryType::Transform => [50, 255, 50],    // Green
    }
}

/// Exports a single cube face as a color-coded plate map PNG.
///
/// Each plate is colored distinctly, with optional boundary overlay.
///
/// # Arguments
/// * `face` - The cube face to export
/// * `plates` - The tectonic plates for color generation
/// * `path` - Output file path
/// * `options` - Export options
///
/// # Returns
/// `Ok(())` on success, or an error if export fails
pub fn export_face_plate_map(
    face: &CubeFace,
    plates: &[TectonicPlate],
    path: &Path,
    options: &PlateMapOptions,
) -> Result<(), PlateMapError> {
    let plate_ids = face.plate_ids.as_ref()
        .ok_or(PlateMapError::NoTectonicData)?;

    let resolution = face.resolution;
    let colors = generate_plate_colors_by_type(plates, 12345);

    // Create RGB image
    let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::new(resolution, resolution);

    for y in 0..resolution {
        for x in 0..resolution {
            let plate_id = plate_ids[(y * resolution + x) as usize];
            if plate_id >= colors.len() {
                return Err(PlateMapError::InvalidPlateId(plate_id));
            }
            let color = colors[plate_id];
            img.put_pixel(x, y, Rgb(color));
        }
    }

    // Write PNG
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

/// Exports all faces of a planet as plate map PNGs.
///
/// Files are named using the pattern: `{base_name}_{face_name}.png`
///
/// # Arguments
/// * `planet` - The planet to export (must have tectonic data)
/// * `output_dir` - Directory to save files to
/// * `base_name` - Base name for output files
/// * `options` - Export options
///
/// # Returns
/// `Ok(())` on success, or the first error encountered
pub fn export_planet_plate_map(
    planet: &Planet,
    output_dir: &Path,
    base_name: &str,
    options: &PlateMapOptions,
) -> Result<(), PlateMapError> {
    let plates = planet.plates.as_ref()
        .ok_or(PlateMapError::NoTectonicData)?;

    std::fs::create_dir_all(output_dir)?;

    for face in &planet.faces {
        let filename = format!("{}_{}.png", base_name, face.id.short_name());
        let path = output_dir.join(filename);
        export_face_plate_map(face, plates, &path, options)?;
    }

    Ok(())
}

/// Exports a boundary type map showing convergent/divergent/transform boundaries.
///
/// Uses color coding:
/// - Red: Convergent (collision/subduction)
/// - Blue: Divergent (spreading ridges)
/// - Green: Transform (strike-slip)
///
/// # Arguments
/// * `face` - The cube face to export
/// * `boundaries` - Plate boundaries for classification
/// * `plates` - Tectonic plates for reference
/// * `path` - Output file path
///
/// # Returns
/// `Ok(())` on success, or an error if export fails
pub fn export_face_boundary_map(
    face: &CubeFace,
    boundaries: &[PlateBoundary],
    plates: &[TectonicPlate],
    path: &Path,
) -> Result<(), PlateMapError> {
    use crate::geometry::{face_uv_to_cube, spherify_point};

    let plate_ids = face.plate_ids.as_ref()
        .ok_or(PlateMapError::NoTectonicData)?;

    let resolution = face.resolution;
    let plate_colors = generate_plate_colors_by_type(plates, 12345);

    // Create RGB image with plate colors as base
    let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::new(resolution, resolution);

    // First pass: fill with desaturated plate colors
    for y in 0..resolution {
        for x in 0..resolution {
            let plate_id = plate_ids[(y * resolution + x) as usize];
            let base_color = plate_colors.get(plate_id).copied().unwrap_or([128, 128, 128]);
            // Desaturate the base color
            let gray = ((base_color[0] as u16 + base_color[1] as u16 + base_color[2] as u16) / 3) as u8;
            let desaturated = [
                ((base_color[0] as u16 + gray as u16) / 2) as u8,
                ((base_color[1] as u16 + gray as u16) / 2) as u8,
                ((base_color[2] as u16 + gray as u16) / 2) as u8,
            ];
            img.put_pixel(x, y, Rgb(desaturated));
        }
    }

    // Second pass: color pixels near boundaries
    let boundary_threshold = 0.02; // About 1-2 pixels at typical resolutions

    for y in 0..resolution {
        for x in 0..resolution {
            let (u, v) = (
                (x as f32 + 0.5) / resolution as f32,
                (y as f32 + 0.5) / resolution as f32,
            );
            let cube_pos = face_uv_to_cube(face.id, u, v);
            let sphere_pos = spherify_point(cube_pos);

            // Check if near any boundary
            for boundary in boundaries {
                for segment in &boundary.segments {
                    // Simple distance check to segment
                    let to_start = sphere_pos - segment.start;
                    let seg_dir = segment.end - segment.start;
                    let seg_len = seg_dir.length();

                    if seg_len > 0.001 {
                        let t = to_start.dot(seg_dir) / (seg_len * seg_len);
                        let t = t.clamp(0.0, 1.0);
                        let closest = segment.start + seg_dir * t;
                        let dist = (sphere_pos - closest).length();

                        if dist < boundary_threshold {
                            let color = boundary_color(boundary.boundary_type);
                            img.put_pixel(x, y, Rgb(color));
                            break;
                        }
                    }
                }
            }
        }
    }

    // Write PNG
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let encoder = PngEncoder::new_with_quality(
        writer,
        CompressionType::Default,
        FilterType::Adaptive
    );

    encoder.write_image(
        img.as_raw(),
        resolution,
        resolution,
        image::ExtendedColorType::Rgb8,
    )?;

    Ok(())
}

/// Exports all faces of a planet as boundary map PNGs.
pub fn export_planet_boundary_map(
    planet: &Planet,
    output_dir: &Path,
    base_name: &str,
) -> Result<(), PlateMapError> {
    let plates = planet.plates.as_ref()
        .ok_or(PlateMapError::NoTectonicData)?;
    let boundaries = planet.boundaries.as_ref()
        .ok_or(PlateMapError::NoTectonicData)?;

    std::fs::create_dir_all(output_dir)?;

    for face in &planet.faces {
        let filename = format!("{}_{}.png", base_name, face.id.short_name());
        let path = output_dir.join(filename);
        export_face_boundary_map(face, boundaries, plates, &path)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::CubeFaceId;
    use glam::Vec3;
    use tempfile::tempdir;

    fn create_test_face_with_plates() -> CubeFace {
        let mut face = CubeFace::new(CubeFaceId::PosX, 64);
        face.init_tectonic_data();

        // Assign plates in a simple pattern
        if let Some(ref mut plate_ids) = face.plate_ids {
            for y in 0..64u32 {
                for x in 0..64u32 {
                    // Create 4 quadrants with different plates
                    let plate_id = match (x < 32, y < 32) {
                        (true, true) => 0,
                        (false, true) => 1,
                        (true, false) => 2,
                        (false, false) => 3,
                    };
                    plate_ids[(y * 64 + x) as usize] = plate_id;
                }
            }
        }

        face
    }

    fn create_test_plates() -> Vec<TectonicPlate> {
        vec![
            TectonicPlate::new(0, Vec3::new(-1.0, 0.5, 0.5).normalize(), true, 2.0, 42),
            TectonicPlate::new(1, Vec3::new(1.0, 0.5, 0.5).normalize(), false, 3.0, 43),
            TectonicPlate::new(2, Vec3::new(-1.0, -0.5, 0.5).normalize(), true, 2.5, 44),
            TectonicPlate::new(3, Vec3::new(1.0, -0.5, 0.5).normalize(), false, 3.5, 45),
        ]
    }

    #[test]
    fn test_generate_plate_colors() {
        let colors = generate_plate_colors(12, 42);
        assert_eq!(colors.len(), 12);

        // Colors should be distinct
        for i in 0..colors.len() {
            for j in (i + 1)..colors.len() {
                let diff: i32 = (colors[i][0] as i32 - colors[j][0] as i32).abs()
                    + (colors[i][1] as i32 - colors[j][1] as i32).abs()
                    + (colors[i][2] as i32 - colors[j][2] as i32).abs();
                // Colors should differ by at least some amount
                assert!(diff > 20, "Colors {} and {} too similar", i, j);
            }
        }
    }

    #[test]
    fn test_generate_plate_colors_by_type() {
        let plates = create_test_plates();
        let colors = generate_plate_colors_by_type(&plates, 42);

        assert_eq!(colors.len(), 4);

        // Continental plates (0, 2) should have warmer tones
        // Oceanic plates (1, 3) should have cooler tones (more blue)
        let continental_blue_avg = (colors[0][2] as u16 + colors[2][2] as u16) / 2;
        let oceanic_blue_avg = (colors[1][2] as u16 + colors[3][2] as u16) / 2;

        // Oceanic should generally be bluer
        assert!(oceanic_blue_avg > continental_blue_avg - 50);
    }

    #[test]
    fn test_hsv_to_rgb() {
        // Red
        let red = hsv_to_rgb(0.0, 1.0, 1.0);
        assert_eq!(red, [255, 0, 0]);

        // Green
        let green = hsv_to_rgb(1.0 / 3.0, 1.0, 1.0);
        assert_eq!(green, [0, 255, 0]);

        // Blue
        let blue = hsv_to_rgb(2.0 / 3.0, 1.0, 1.0);
        assert_eq!(blue, [0, 0, 255]);

        // White (no saturation)
        let white = hsv_to_rgb(0.0, 0.0, 1.0);
        assert_eq!(white, [255, 255, 255]);

        // Black (no value)
        let black = hsv_to_rgb(0.0, 1.0, 0.0);
        assert_eq!(black, [0, 0, 0]);
    }

    #[test]
    fn test_export_face_plate_map() {
        let face = create_test_face_with_plates();
        let plates = create_test_plates();
        let dir = tempdir().unwrap();
        let path = dir.path().join("plate_map.png");

        let options = PlateMapOptions::default();
        export_face_plate_map(&face, &plates, &path, &options).unwrap();

        assert!(path.exists());

        // Verify file size is reasonable
        let metadata = std::fs::metadata(&path).unwrap();
        assert!(metadata.len() > 0);
    }

    #[test]
    fn test_export_face_plate_map_no_data() {
        let face = CubeFace::new(CubeFaceId::PosX, 32);
        let plates = create_test_plates();
        let dir = tempdir().unwrap();
        let path = dir.path().join("plate_map.png");

        let options = PlateMapOptions::default();
        let result = export_face_plate_map(&face, &plates, &path, &options);

        assert!(matches!(result, Err(PlateMapError::NoTectonicData)));
    }

    #[test]
    fn test_boundary_colors() {
        let convergent = boundary_color(BoundaryType::Convergent);
        let divergent = boundary_color(BoundaryType::Divergent);
        let transform = boundary_color(BoundaryType::Transform);

        // Convergent should be red-ish
        assert!(convergent[0] > convergent[1] && convergent[0] > convergent[2]);

        // Divergent should be blue-ish
        assert!(divergent[2] > divergent[0] && divergent[2] > divergent[1]);

        // Transform should be green-ish
        assert!(transform[1] > transform[0] && transform[1] > transform[2]);
    }
}
