//! Erosion configuration.

use serde::{Deserialize, Serialize};

/// Which backend to use for hydraulic/thermal erosion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErosionBackend {
    /// Prefer GPU; if GPU init fails, fall back to CPU.
    Auto,
    /// Require GPU (fail the stage if unavailable).
    GpuOnly,
    /// Force CPU implementation.
    CpuOnly,
}

impl Default for ErosionBackend {
    fn default() -> Self {
        Self::Auto
    }
}

/// How depressions “drain” on a closed planetary surface for depression filling.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OutletModel {
    /// Treat all cells with height <= `sea_level` as open outlets.
    SeaLevel { sea_level: f32 },
}

impl Default for OutletModel {
    fn default() -> Self {
        // Earth-ish baseline: sea level at 0.0 (your heights are currently unit-ish / km-ish).
        Self::SeaLevel { sea_level: 0.0 }
    }
}

/// Parameters for Phase 3 erosion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErosionConfig {
    /// Which backend to use for hydraulic+thermal erosion.
    pub backend: ErosionBackend,
    /// Number of hydraulic erosion timesteps.
    pub hydraulic_steps: u32,
    /// Rainfall per step (water units per cell per step).
    pub rainfall: f32,
    /// Evaporation factor per step (0-1).
    pub evaporation: f32,

    /// Erosion rate (Ks).
    pub erosion_rate: f32,
    /// Deposition rate (Kd).
    pub deposition_rate: f32,
    /// Sediment capacity factor (Kc).
    pub sediment_capacity: f32,

    /// Thermal erosion iterations.
    pub thermal_iterations: u32,
    /// Angle of repose (radians). Typical: 30-45 degrees.
    pub angle_of_repose_rad: f32,
    /// Fraction of excess slope to move per iteration.
    pub thermal_strength: f32,

    /// Depression filling/outlet behavior.
    pub outlet_model: OutletModel,

    /// River extraction threshold (minimum contributing cells).
    pub river_accum_threshold: u32,

    /// If true, keep intermediate maps on `CubeFace` for exporting/inspection.
    pub keep_intermediates: bool,

    /// If true, compute and store a net deposition/erosion map (per-cell height delta)
    /// caused by the erosion backend (hydraulic + thermal), excluding depression filling.
    pub track_deposition: bool,
}

impl Default for ErosionConfig {
    fn default() -> Self {
        Self {
            backend: ErosionBackend::default(),
            hydraulic_steps: 200,
            rainfall: 0.01,
            evaporation: 0.02,

            erosion_rate: 0.02,
            deposition_rate: 0.02,
            sediment_capacity: 0.05,

            thermal_iterations: 150,
            angle_of_repose_rad: 35_f32.to_radians(),
            thermal_strength: 0.25,

            outlet_model: OutletModel::default(),

            river_accum_threshold: 500,
            keep_intermediates: false,
            track_deposition: false,
        }
    }
}

