//! Heightmap generation using fractal noise.

use rayon::prelude::*;
use crate::geometry::{face_uv_to_cube, cube_to_sphere};
use crate::noise::{FractalNoiseConfig, sample_fractal_noise};
use super::planet::{Planet, CubeFace};

/// Generates heightmaps for all faces of a planet using fractal noise.
///
/// This function uses parallel processing (rayon) to efficiently generate
/// terrain for all six cube faces simultaneously.
///
/// # Arguments
/// * `planet` - The planet to generate heightmaps for
/// * `config` - Fractal noise configuration
pub fn generate_heightmap(planet: &mut Planet, config: &FractalNoiseConfig) {
    // Process all faces in parallel
    planet.faces.par_iter_mut().for_each(|face| {
        generate_face_heightmap(face, config);
    });
}

/// Generates a heightmap for a single cube face.
///
/// # Arguments
/// * `face` - The cube face to generate heights for
/// * `config` - Fractal noise configuration
pub fn generate_face_heightmap(face: &mut CubeFace, config: &FractalNoiseConfig) {
    let resolution = face.resolution;
    let face_id = face.id;

    // Process heights in parallel using rayon
    face.heights.par_iter_mut().enumerate().for_each(|(i, height)| {
        let x = (i as u32) % resolution;
        let y = (i as u32) / resolution;

        // Convert pixel to UV coordinates
        let u = (x as f32 + 0.5) / resolution as f32;
        let v = (y as f32 + 0.5) / resolution as f32;

        // Convert to sphere position and sample noise
        let cube_point = face_uv_to_cube(face_id, u, v);
        let sphere_point = cube_to_sphere(cube_point);
        *height = sample_fractal_noise(sphere_point, config);
    });
}

/// Generates heightmaps with a simple continent/ocean bias.
///
/// This adds a base layer that creates larger landmasses and ocean basins,
/// more suitable for Earth-like planets.
pub fn generate_heightmap_with_continents(planet: &mut Planet, config: &FractalNoiseConfig) {
    // First, generate the base fractal terrain
    generate_heightmap(planet, config);

    // Create a low-frequency continent mask
    let continent_config = FractalNoiseConfig {
        octaves: 3,
        frequency: 0.5,
        lacunarity: 2.0,
        persistence: 0.6,
        seed: config.seed.wrapping_add(999),
    };

    // Apply continent bias to each face
    for face in &mut planet.faces {
        let resolution = face.resolution;
        let face_id = face.id;

        face.heights.par_iter_mut().enumerate().for_each(|(i, height)| {
            let x = (i as u32) % resolution;
            let y = (i as u32) / resolution;

            let u = (x as f32 + 0.5) / resolution as f32;
            let v = (y as f32 + 0.5) / resolution as f32;

            let cube_point = face_uv_to_cube(face_id, u, v);
            let sphere_point = cube_to_sphere(cube_point);

            // Get continent mask value
            let continent_mask = sample_fractal_noise(sphere_point, &continent_config);

            // Blend: positive continent mask = land (raise), negative = ocean (lower)
            let bias = continent_mask * 0.3;
            *height = (*height * 0.7) + bias;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::CubeFaceId;

    #[test]
    fn test_generate_heightmap() {
        let mut planet = Planet::new(64, 42, 6371.0);
        let config = FractalNoiseConfig::default();

        generate_heightmap(&mut planet, &config);

        // Check that heights were generated (not all zeros)
        let has_non_zero = planet.faces.iter()
            .any(|face| face.heights.iter().any(|&h| h != 0.0));
        assert!(has_non_zero, "Heightmap should have non-zero values");

        // Check reasonable range
        let (min, max) = planet.height_range();
        assert!(min >= -2.0 && max <= 2.0, "Heights should be in reasonable range");
    }

    #[test]
    fn test_generate_face_heightmap() {
        let mut face = CubeFace::new(CubeFaceId::PosX, 32);
        let config = FractalNoiseConfig::with_seed(123);

        generate_face_heightmap(&mut face, &config);

        // Verify heights were set
        assert!(face.heights.iter().any(|&h| h != 0.0));
    }

    #[test]
    fn test_heightmap_reproducibility() {
        let config = FractalNoiseConfig::with_seed(999);

        let mut planet1 = Planet::new(32, 42, 6371.0);
        let mut planet2 = Planet::new(32, 42, 6371.0);

        generate_heightmap(&mut planet1, &config);
        generate_heightmap(&mut planet2, &config);

        // Heights should be identical
        for (f1, f2) in planet1.faces.iter().zip(planet2.faces.iter()) {
            for (h1, h2) in f1.heights.iter().zip(f2.heights.iter()) {
                assert_eq!(h1, h2, "Same configuration should produce identical heights");
            }
        }
    }

    #[test]
    fn test_continent_generation() {
        let mut planet = Planet::new(64, 42, 6371.0);
        let config = FractalNoiseConfig::earth_like(42);

        generate_heightmap_with_continents(&mut planet, &config);

        // Check that heights were generated
        let (min, max) = planet.height_range();
        assert!(min < max, "Should have height variation");
    }
}
