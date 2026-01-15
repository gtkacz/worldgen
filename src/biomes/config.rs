//! Biome configuration for Phase 5.

/// Configuration for biome classification and derived map generation.
///
/// Notes:
/// - Heights are in km (matches `CubeFace.heights` usage).
/// - Temperatures are °C.
/// - Precipitation is mm/year (as produced by `ClimateStage`).
#[derive(Debug, Clone)]
pub struct BiomeConfig {
    /// Sea level (km). Pixels at or below this are treated as ocean and masked out.
    pub sea_level: f32,

    /// Seed used for biome boundary jitter (ecotones).
    pub seed: u64,

    // --- Ecotone jitter (noise-perturbed boundaries) ---
    /// Jitter amplitude applied to temperature (°C).
    pub jitter_temp_c: f32,
    /// Jitter amplitude applied to precipitation (mm/year).
    pub jitter_precip_mm: f32,
    /// Noise frequency for jitter sampling.
    pub jitter_frequency: f32,

    // --- Derived maps ---
    /// Scale factor mapping ruggedness (km deltas) to roughness [0,1].
    pub roughness_scale: f32,
    /// Strength of river proximity boost applied to vegetation.
    pub river_veg_boost: f32,

    /// Temperature at/above which snow overlay is 0.
    pub snow_melt_temp_c: f32,
    /// Temperature at/below which snow overlay is 1.
    pub snow_freeze_temp_c: f32,
}

impl Default for BiomeConfig {
    fn default() -> Self {
        Self {
            sea_level: 0.0,
            seed: 0,

            jitter_temp_c: 1.5,
            jitter_precip_mm: 120.0,
            jitter_frequency: 1.25,

            roughness_scale: 0.75,
            river_veg_boost: 0.25,

            snow_melt_temp_c: 2.0,
            snow_freeze_temp_c: -12.0,
        }
    }
}

