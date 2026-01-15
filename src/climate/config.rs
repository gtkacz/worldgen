//! Climate configuration parameters.

/// Configuration for the Phase 4 climate simulation.
///
/// Units:
/// - temperatures: °C
/// - elevation: km (matches `CubeFace.heights` usage in this project)
/// - distances: km
/// - precipitation: mm (relative units; exported consistently)
#[derive(Debug, Clone)]
pub struct ClimateConfig {
    // Core reference parameters
    pub sea_level: f32,
    pub axial_tilt_deg: f32,

    // Season model
    pub months: u8,        // expected 12
    pub season_phase: f32, // fraction of year [0,1)

    // Temperature model
    pub equator_temp_c: f32,
    pub pole_temp_c: f32,
    pub lapse_rate_c_per_km: f32,
    pub maritime_buffer_km: f32,
    pub ocean_temp_c: f32,
    pub seasonality_c: f32, // baseline seasonal amplitude at poles for ~23.5° tilt

    // Wind model
    pub itcz_shift_deg: f32,
    pub meridional_strength: f32, // 0..1, mixes north/south into zonal flow

    // Moisture/precipitation model
    pub ocean_evap_base_mm: f32,
    pub rainout_rate: f32,
    pub orographic_scale: f32,
    pub iterations: u32,
}

impl Default for ClimateConfig {
    fn default() -> Self {
        // Tuned for “believable” outputs rather than strict physical realism.
        Self {
            sea_level: 0.0,
            axial_tilt_deg: 23.44,
            months: 12,
            season_phase: 0.0,

            equator_temp_c: 30.0,
            pole_temp_c: -20.0,
            lapse_rate_c_per_km: 6.5,
            maritime_buffer_km: 450.0,
            ocean_temp_c: 27.0,
            seasonality_c: 25.0,

            itcz_shift_deg: 10.0,
            meridional_strength: 0.25,

            ocean_evap_base_mm: 6.0,
            rainout_rate: 0.06,
            orographic_scale: 0.6,
            iterations: 64,
        }
    }
}

impl ClimateConfig {
    pub fn earth_like() -> Self {
        Self::default()
    }
}

