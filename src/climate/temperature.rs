//! Monthly temperature model.

use super::util::month_phase_sin;
use super::ClimateConfig;

/// Compute monthly temperature (°C) for a point given latitude, elevation, and coast distance.
pub fn temperature_c(
    latitude_rad: f32,
    elevation_km: f32,
    is_ocean: bool,
    coast_distance_km: f32,
    month_idx: u8,
    cfg: &ClimateConfig,
) -> f32 {
    let abs_lat = latitude_rad.abs();
    let t = (abs_lat / (std::f32::consts::FRAC_PI_2)).clamp(0.0, 1.0);

    // Latitudinal gradient. Nonlinear curve gives larger polar contrast.
    let lat_w = t.powf(1.15);
    let baseline = cfg.equator_temp_c * (1.0 - lat_w) + cfg.pole_temp_c * lat_w;

    // Seasonal component: depends on axial tilt and hemisphere.
    // month_phase_sin is +1 at NH summer peak (by convention); lat sign flips seasons across hemispheres.
    let tilt_scale = (cfg.axial_tilt_deg / 23.44).clamp(0.0, 2.0);
    let s = month_phase_sin(month_idx, cfg.months, cfg.season_phase);
    let seasonal = s * cfg.seasonality_c * tilt_scale * latitude_rad.sin();

    // Ocean is moderated; land gets full seasonal swing.
    let seasonal = if is_ocean { seasonal * 0.35 } else { seasonal };

    // Altitude lapse relative to sea level.
    let above_sea_km = (elevation_km - cfg.sea_level).max(0.0);
    let lapse = cfg.lapse_rate_c_per_km * above_sea_km;

    let mut temp = baseline + seasonal - lapse;

    // Maritime buffering for land: blend toward an “ocean baseline”.
    if !is_ocean && coast_distance_km.is_finite() {
        let w = (-coast_distance_km / cfg.maritime_buffer_km.max(1.0)).exp().clamp(0.0, 1.0);
        let ocean_baseline = baseline * 0.6 + cfg.ocean_temp_c * 0.4;
        temp = temp * (1.0 - w) + ocean_baseline * w;
    }

    temp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equator_is_warmer_than_pole_in_annual_baseline() {
        let cfg = ClimateConfig::default();
        let equator = temperature_c(0.0, 0.0, false, 1e9, 0, &cfg);
        let pole = temperature_c(std::f32::consts::FRAC_PI_2, 0.0, false, 1e9, 0, &cfg);
        assert!(equator > pole);
    }

    #[test]
    fn higher_elevation_is_colder_by_lapse_rate() {
        let mut cfg = ClimateConfig::default();
        cfg.sea_level = 0.0;
        // Make buffering negligible for deterministic lapse behavior.
        let coast_far = 1e9;

        let lat = 0.25; // ~14°
        let t0 = temperature_c(lat, 0.0, false, coast_far, 0, &cfg);
        let t1 = temperature_c(lat, 1.0, false, coast_far, 0, &cfg);
        let expected_drop = cfg.lapse_rate_c_per_km;
        let actual_drop = t0 - t1;
        assert!(
            (actual_drop - expected_drop).abs() < 1e-3,
            "expected drop ~{}, got {}",
            expected_drop,
            actual_drop
        );
    }
}

