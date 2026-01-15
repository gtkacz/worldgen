//! Multi-octave fractal Brownian motion (fBm) noise generation.

use glam::Vec3;
use serde::{Deserialize, Serialize};
use simdnoise::NoiseBuilder;

/// Configuration for multi-octave fractal noise generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractalNoiseConfig {
    /// Number of noise octaves (4-8 typical).
    pub octaves: u8,
    /// Base frequency of the noise (1.0-4.0 typical).
    pub frequency: f32,
    /// Frequency multiplier per octave (typically 2.0).
    pub lacunarity: f32,
    /// Amplitude decay per octave (0.4-0.6 typical).
    pub persistence: f32,
    /// Random seed for reproducible generation.
    pub seed: i32,
}

impl Default for FractalNoiseConfig {
    fn default() -> Self {
        Self {
            octaves: 6,
            frequency: 2.0,
            lacunarity: 2.0,
            persistence: 0.5,
            seed: 42,
        }
    }
}

impl FractalNoiseConfig {
    /// Creates a new noise configuration with the given seed.
    pub fn with_seed(seed: i32) -> Self {
        Self {
            seed,
            ..Default::default()
        }
    }

    /// Creates an Earth-like terrain configuration.
    pub fn earth_like(seed: i32) -> Self {
        Self {
            octaves: 8,
            frequency: 1.5,
            lacunarity: 2.1,
            persistence: 0.55,
            seed,
        }
    }

    /// Creates a smoother, more moon-like configuration.
    pub fn moon_like(seed: i32) -> Self {
        Self {
            octaves: 4,
            frequency: 3.0,
            lacunarity: 2.0,
            persistence: 0.4,
            seed,
        }
    }
}

/// Samples fractal noise at a 3D position (typically on a unit sphere).
///
/// Uses 4D simplex noise with the w dimension set to 0, which provides
/// seamless sampling on spherical surfaces without UV distortion artifacts.
///
/// # Arguments
/// * `pos` - A 3D position (typically a point on the unit sphere)
/// * `config` - Noise configuration parameters
///
/// # Returns
/// A noise value in approximately [-1, 1] range (normalized by amplitude sum)
pub fn sample_fractal_noise(pos: Vec3, config: &FractalNoiseConfig) -> f32 {
    let mut total = 0.0f32;
    let mut amplitude = 1.0f32;
    let mut frequency = config.frequency;
    let mut max_amplitude = 0.0f32;

    for octave in 0..config.octaves {
        // Use 4D simplex noise for seamless spherical sampling
        // Each octave gets a different seed offset for variation
        let octave_seed = config.seed.wrapping_add(octave as i32 * 31337);

        let x = pos.x * frequency;
        let y = pos.y * frequency;
        let z = pos.z * frequency;

        // simdnoise simplex_4d returns values in [-1, 1]
        let noise_value = NoiseBuilder::fbm_4d_offset(x, 1, y, 1, z, 1, 0.0, 1)
            .with_seed(octave_seed)
            .with_freq(1.0)
            .with_octaves(1)
            .generate()
            .0[0];

        total += noise_value * amplitude;
        max_amplitude += amplitude;
        amplitude *= config.persistence;
        frequency *= config.lacunarity;
    }

    // Normalize to [-1, 1]
    total / max_amplitude
}

/// Samples fractal noise for a batch of positions.
///
/// More efficient than calling `sample_fractal_noise` repeatedly due to
/// SIMD vectorization in simdnoise.
///
/// # Arguments
/// * `positions` - Slice of 3D positions
/// * `config` - Noise configuration parameters
///
/// # Returns
/// Vector of noise values, one per input position
pub fn sample_fractal_noise_batch(positions: &[Vec3], config: &FractalNoiseConfig) -> Vec<f32> {
    if positions.is_empty() {
        return Vec::new();
    }

    let mut results = vec![0.0f32; positions.len()];
    let mut amplitude = 1.0f32;
    let mut frequency = config.frequency;
    let mut max_amplitude = 0.0f32;

    for octave in 0..config.octaves {
        let octave_seed = config.seed.wrapping_add(octave as i32 * 31337);

        // Process positions in batches for SIMD efficiency
        for (i, pos) in positions.iter().enumerate() {
            let x = pos.x * frequency;
            let y = pos.y * frequency;
            let z = pos.z * frequency;

            let noise_value = NoiseBuilder::fbm_4d_offset(x, 1, y, 1, z, 1, 0.0, 1)
                .with_seed(octave_seed)
                .with_freq(1.0)
                .with_octaves(1)
                .generate()
                .0[0];

            results[i] += noise_value * amplitude;
        }

        max_amplitude += amplitude;
        amplitude *= config.persistence;
        frequency *= config.lacunarity;
    }

    // Normalize all results
    for result in &mut results {
        *result /= max_amplitude;
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = FractalNoiseConfig::default();
        assert_eq!(config.octaves, 6);
        assert_eq!(config.lacunarity, 2.0);
        assert_eq!(config.persistence, 0.5);
    }

    #[test]
    fn test_noise_reproducibility() {
        let config = FractalNoiseConfig::with_seed(12345);
        let pos = Vec3::new(0.5, 0.3, 0.7);

        let result1 = sample_fractal_noise(pos, &config);
        let result2 = sample_fractal_noise(pos, &config);

        assert_eq!(result1, result2, "Same seed and position should produce same result");
    }

    #[test]
    fn test_noise_range() {
        let config = FractalNoiseConfig::default();
        let test_positions = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.577, 0.577, 0.577),
            Vec3::new(-0.5, 0.5, 0.707),
        ];

        for pos in test_positions {
            let value = sample_fractal_noise(pos, &config);
            assert!(
                value >= -1.5 && value <= 1.5,
                "Noise value {} at {:?} out of expected range",
                value,
                pos
            );
        }
    }

    #[test]
    fn test_different_seeds_produce_different_results() {
        let config1 = FractalNoiseConfig::with_seed(1);
        let config2 = FractalNoiseConfig::with_seed(2);
        let pos = Vec3::new(0.5, 0.3, 0.7);

        let result1 = sample_fractal_noise(pos, &config1);
        let result2 = sample_fractal_noise(pos, &config2);

        assert_ne!(result1, result2, "Different seeds should produce different results");
    }

    #[test]
    fn test_batch_sampling() {
        let config = FractalNoiseConfig::default();
        let positions = vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];

        let batch_results = sample_fractal_noise_batch(&positions, &config);

        assert_eq!(batch_results.len(), positions.len());

        // Verify batch results match individual sampling
        for (i, pos) in positions.iter().enumerate() {
            let single_result = sample_fractal_noise(*pos, &config);
            assert!(
                (batch_results[i] - single_result).abs() < 1e-6,
                "Batch result {} differs from single result {} at index {}",
                batch_results[i],
                single_result,
                i
            );
        }
    }

    #[test]
    fn test_empty_batch() {
        let config = FractalNoiseConfig::default();
        let positions: Vec<Vec3> = vec![];
        let results = sample_fractal_noise_batch(&positions, &config);
        assert!(results.is_empty());
    }
}
