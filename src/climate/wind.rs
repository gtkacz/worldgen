//! Simplified three-cell wind model.

use glam::Vec3;

use super::util::{local_tangent_basis, month_phase_sin};
use super::ClimateConfig;

/// Returns a tangent unit wind direction at sphere point `p` for the given month index [0..months).
///
/// This uses a very simplified three-cell approximation:
/// - 0–30°: easterlies (westward)
/// - 30–60°: westerlies (eastward)
/// - 60–90°: polar easterlies (westward)
///
/// Adds a small meridional component that converges toward the seasonally shifted ITCZ.
pub fn prevailing_wind_tangent(p: Vec3, latitude_rad: f32, month_idx: u8, cfg: &ClimateConfig) -> Vec3 {
    let (east, north) = local_tangent_basis(p);
    let lat_deg = latitude_rad.to_degrees();

    // Seasonal ITCZ shift (degrees).
    let s = month_phase_sin(month_idx, cfg.months, cfg.season_phase);
    let itcz = cfg.itcz_shift_deg * s;

    // Use effective latitude relative to ITCZ for low-lat convergence.
    let rel_lat = lat_deg - itcz;
    let abs_lat = rel_lat.abs();

    // Zonal direction by latitude band.
    let zonal = if abs_lat < 30.0 {
        -east // easterlies -> westward
    } else if abs_lat < 60.0 {
        east // westerlies -> eastward
    } else {
        -east // polar easterlies
    };

    // Meridional component: converge toward ITCZ within the tropics.
    let toward_itcz = if rel_lat >= 0.0 { -north } else { north };
    let tropics_w = (1.0 - (abs_lat / 30.0).clamp(0.0, 1.0)).powf(1.5);
    let m = cfg.meridional_strength * tropics_w;

    (zonal * (1.0 - m) + toward_itcz * m).normalize_or_zero()
}

