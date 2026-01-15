//! Configuration for tectonic simulation.

use serde::{Deserialize, Serialize};

/// Configuration parameters for tectonic plate simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TectonicConfig {
    /// Number of tectonic plates (8-15 typical for Earth-like).
    pub num_plates: usize,
    /// Fraction of surface area that is continental (0.3-0.4 for Earth-like).
    pub continental_fraction: f32,
    /// Number of Lloyd's relaxation iterations for plate center distribution.
    pub lloyd_iterations: usize,

    /// Base velocity scale in cm/year equivalent.
    pub velocity_scale: f32,
    /// Random variation in plate velocities (0.0-1.0).
    pub velocity_randomness: f32,

    /// Scale factor for subduction-induced uplift.
    pub subduction_uplift_scale: f32,
    /// Scale factor for trench depth.
    pub trench_depth_scale: f32,
    /// Base distance from trench to volcanic arc (km).
    pub arc_distance_base: f32,

    /// Scale factor for collision-induced uplift.
    pub collision_uplift_scale: f32,
    /// Base width of mountain ranges (km).
    pub mountain_width_base: f32,

    /// Depth of ridge crest below sea level (km, negative).
    pub ridge_crest_depth: f32,
    /// Rate of thermal subsidence (km/sqrt(Ma)).
    pub thermal_subsidence_rate: f32,

    /// Random seed for reproducible generation.
    pub seed: u64,
}

impl Default for TectonicConfig {
    fn default() -> Self {
        Self {
            num_plates: 12,
            continental_fraction: 0.35,
            lloyd_iterations: 3,
            velocity_scale: 5.0,
            velocity_randomness: 0.3,
            subduction_uplift_scale: 1.0,
            trench_depth_scale: 1.0,
            arc_distance_base: 150.0,
            collision_uplift_scale: 1.0,
            mountain_width_base: 200.0,
            ridge_crest_depth: -2.5,
            thermal_subsidence_rate: 0.35,
            seed: 42,
        }
    }
}

impl TectonicConfig {
    /// Creates a configuration suitable for Earth-like planets.
    pub fn earth_like(seed: u64) -> Self {
        Self {
            seed,
            ..Default::default()
        }
    }

    /// Creates a configuration with many small plates (more active tectonics).
    pub fn active(seed: u64) -> Self {
        Self {
            num_plates: 20,
            velocity_scale: 8.0,
            velocity_randomness: 0.4,
            subduction_uplift_scale: 1.2,
            collision_uplift_scale: 1.3,
            seed,
            ..Default::default()
        }
    }

    /// Creates a configuration with few large plates (more stable tectonics).
    pub fn stable(seed: u64) -> Self {
        Self {
            num_plates: 6,
            velocity_scale: 2.0,
            velocity_randomness: 0.2,
            subduction_uplift_scale: 0.8,
            collision_uplift_scale: 0.7,
            seed,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TectonicConfig::default();
        assert_eq!(config.num_plates, 12);
        assert!((config.continental_fraction - 0.35).abs() < 0.01);
    }

    #[test]
    fn test_earth_like_config() {
        let config = TectonicConfig::earth_like(123);
        assert_eq!(config.seed, 123);
        assert_eq!(config.num_plates, 12);
    }

    #[test]
    fn test_active_config() {
        let config = TectonicConfig::active(456);
        assert_eq!(config.num_plates, 20);
        assert!(config.velocity_scale > TectonicConfig::default().velocity_scale);
    }

    #[test]
    fn test_stable_config() {
        let config = TectonicConfig::stable(789);
        assert_eq!(config.num_plates, 6);
        assert!(config.velocity_scale < TectonicConfig::default().velocity_scale);
    }
}
