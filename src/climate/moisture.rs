//! Moisture advection and precipitation model.

use glam::Vec3;
use rayon::prelude::*;

use crate::geometry::sphere_to_face_uv;
use crate::terrain::CubeFace;

use super::util::great_circle_step;
use super::{temperature, wind, ClimateConfig};

#[derive(Debug, Clone)]
pub struct MonthlyClimate {
    /// Temperature (°C), length `6*res*res`.
    pub temperature_c: Vec<f32>,
    /// Precipitation (mm/month), length `6*res*res`.
    pub precipitation_mm: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct ClimatePrecompute {
    pub res: u32,
    /// Unit sphere positions per pixel (flattened), length `6*res*res`.
    pub sphere_points: Vec<Vec3>,
    /// Latitude (radians) per pixel (flattened), length `6*res*res`.
    pub lat_rad: Vec<f32>,
}

pub fn precompute_lat_lon(res: u32, faces: &[CubeFace; 6]) -> ClimatePrecompute {
    let per_face = (res * res) as usize;
    let total = per_face * 6;
    let mut sphere_points = vec![Vec3::ZERO; total];
    let mut lat_rad = vec![0.0f32; total];

    for face in faces {
        let face_id = face.id;
        let fi = face_id.index();
        let base = fi * per_face;
        for i in 0..per_face {
            let x = (i as u32) % res;
            let y = (i as u32) / res;
            let (u, v) = face.pixel_to_uv(x, y);
            let p = crate::geometry::cube_to_sphere(crate::geometry::face_uv_to_cube(face_id, u, v));
            sphere_points[base + i] = p;
            lat_rad[base + i] = p.y.asin().clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
        }
    }

    ClimatePrecompute { res, sphere_points, lat_rad }
}

/// Compute monthly climate fields at full resolution for the entire planet.
///
/// Inputs are flattened arrays in face order.
pub fn compute_month(
    cfg: &ClimateConfig,
    month_idx: u8,
    planet_radius_km: f32,
    heights_km: &[f32],
    coast_distance_km: &[f32],
    pre: &ClimatePrecompute,
) -> MonthlyClimate {
    let res = pre.res;
    let per_face = (res * res) as usize;
    let total = per_face * 6;
    assert_eq!(heights_km.len(), total);
    assert_eq!(coast_distance_km.len(), total);

    // Temperature field (computed once).
    let temperature_c: Vec<f32> = (0..total)
        .into_par_iter()
        .map(|i| {
            let elev = heights_km[i];
            let is_ocean = elev <= cfg.sea_level;
            temperature::temperature_c(
                pre.lat_rad[i],
                elev,
                is_ocean,
                coast_distance_km[i],
                month_idx,
                cfg,
            )
        })
        .collect();

    // Precompute upwind + downwind mapping per pixel for this month.
    // Use a step angle similar to one pixel spacing on a 90° face.
    let step_angle = std::f32::consts::FRAC_PI_2 / res.max(1) as f32;

    let upwind_idx: Vec<usize> = (0..total)
        .into_par_iter()
        .map(|i| {
            let p = pre.sphere_points[i];
            let lat = pre.lat_rad[i];
            let w = wind::prevailing_wind_tangent(p, lat, month_idx, cfg);
            let p_up = great_circle_step(p, -w, step_angle);
            face_uv_to_global_index(res, p_up)
        })
        .collect();

    let downwind_idx: Vec<usize> = (0..total)
        .into_par_iter()
        .map(|i| {
            let p = pre.sphere_points[i];
            let lat = pre.lat_rad[i];
            let w = wind::prevailing_wind_tangent(p, lat, month_idx, cfg);
            let p_dn = great_circle_step(p, w, step_angle);
            face_uv_to_global_index(res, p_dn)
        })
        .collect();

    // Iterative Jacobi moisture advection.
    let mut moisture_prev = vec![0.0f32; total];
    let mut moisture_next = vec![0.0f32; total];
    let mut precip_accum = vec![0.0f32; total];

    let iters = cfg.iterations.max(1);
    for _ in 0..iters {
        // Compute next moisture and precip for this substep.
        moisture_next
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, m_out)| {
                let src = upwind_idx[i];
                let mut m = moisture_prev[src];

                let elev = heights_km[i];
                let is_ocean = elev <= cfg.sea_level;
                if is_ocean {
                    let t = temperature_c[i];
                    let evap_w = ((t / 30.0)).clamp(0.0, 1.0);
                    m += cfg.ocean_evap_base_mm * evap_w;
                }

                // Orographic lift proxy using downwind height difference.
                let dn = downwind_idx[i];
                let lift = (heights_km[dn] - elev).max(0.0);

                let mut precip = cfg.rainout_rate.clamp(0.0, 1.0) * m;
                precip *= 1.0 + cfg.orographic_scale.max(0.0) * lift;
                precip = precip.min(m).max(0.0);

                // Store substep precipitation by atomic-free two-phase write:
                // precip_accum is updated outside this loop (see below).
                *m_out = (m - precip).max(0.0);
            });

        // Accumulate precip for this step (separate pass to avoid contention).
        precip_accum
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, p_out)| {
                let src = upwind_idx[i];
                let mut m = moisture_prev[src];
                let elev = heights_km[i];
                let is_ocean = elev <= cfg.sea_level;
                if is_ocean {
                    let t = temperature_c[i];
                    let evap_w = ((t / 30.0)).clamp(0.0, 1.0);
                    m += cfg.ocean_evap_base_mm * evap_w;
                }
                let dn = downwind_idx[i];
                let lift = (heights_km[dn] - elev).max(0.0);

                let mut precip = cfg.rainout_rate.clamp(0.0, 1.0) * m;
                precip *= 1.0 + cfg.orographic_scale.max(0.0) * lift;
                precip = precip.min(m).max(0.0);

                *p_out += precip;
            });

        std::mem::swap(&mut moisture_prev, &mut moisture_next);
    }

    // Convert accumulated substep precip to monthly precip; here we treat iterations as sub-month steps,
    // so total accumulated precip is already the monthly total in mm-units.
    let precipitation_mm = precip_accum;

    let _ = planet_radius_km; // reserved for future scaling

    MonthlyClimate { temperature_c, precipitation_mm }
}

fn face_uv_to_global_index(resolution: u32, sphere_pos: Vec3) -> usize {
    let (face, u, v) = sphere_to_face_uv(sphere_pos);
    // Convert UV to nearest pixel on that face.
    let x = ((u * resolution as f32) as u32).min(resolution - 1);
    let y = ((v * resolution as f32) as u32).min(resolution - 1);
    let per_face = (resolution * resolution) as usize;
    face.index() * per_face + (y * resolution + x) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::Planet;

    #[test]
    fn monthly_precip_is_non_negative() {
        let planet = Planet::new(16, 1, 6371.0);
        let cfg = ClimateConfig { iterations: 8, ..Default::default() };

        let res = planet.resolution();
        let per_face = (res * res) as usize;
        let total = per_face * 6;
        let heights = vec![0.0f32; total];
        let coast = vec![0.0f32; total];
        let pre = precompute_lat_lon(res, &planet.faces);

        let out = compute_month(&cfg, 0, planet.radius, &heights, &coast, &pre);
        assert!(out.precipitation_mm.iter().all(|&p| p >= 0.0));
    }

    #[test]
    fn orographic_lift_increases_local_precipitation() {
        let planet = Planet::new(32, 1, 6371.0);
        let res = planet.resolution();
        let per_face = (res * res) as usize;
        let total = per_face * 6;

        let mut cfg = ClimateConfig::default();
        cfg.iterations = 6;
        cfg.rainout_rate = 0.25;
        cfg.orographic_scale = 2.0;
        // Sea level at 0 so negative elevations are ocean moisture sources.
        cfg.sea_level = 0.0;

        let pre = precompute_lat_lon(res, &planet.faces);

        // Pick a target pixel near the equator to ensure we're in the tropical wind band.
        let (i, _lat) = pre
            .lat_rad
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .unwrap();

        // Compute downwind index using the same logic as the solver.
        let step_angle = std::f32::consts::FRAC_PI_2 / res.max(1) as f32;
        let p = pre.sphere_points[i];
        let lat = pre.lat_rad[i];
        let w = super::super::wind::prevailing_wind_tangent(p, lat, 0, &cfg);
        let p_dn = super::super::util::great_circle_step(p, w, step_angle);
        let (face, u, v) = crate::geometry::sphere_to_face_uv(p_dn);
        let x = ((u * res as f32) as u32).min(res - 1);
        let y = ((v * res as f32) as u32).min(res - 1);
        let dn = face.index() * per_face + (y * res + x) as usize;

        // Baseline: flat ocean everywhere.
        let mut heights_flat = vec![-0.5f32; total];
        let coast = vec![0.0f32; total];
        let base = compute_month(&cfg, 0, planet.radius, &heights_flat, &coast, &pre);

        // Ridge case: downwind cell is higher, creating positive lift at i.
        let mut heights_ridge = heights_flat.clone();
        heights_ridge[dn] = 1.0;
        let ridge = compute_month(&cfg, 0, planet.radius, &heights_ridge, &coast, &pre);

        assert!(
            ridge.precipitation_mm[i] > base.precipitation_mm[i],
            "expected orographic lift to increase precip at target cell (base={}, ridge={})",
            base.precipitation_mm[i],
            ridge.precipitation_mm[i]
        );
    }
}

