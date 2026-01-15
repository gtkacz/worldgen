//! Phase 5: Biome classification + derived map generation.
//!
//! Consumes Phase 4 climate summaries (annual mean temperature, annual precipitation,
//! min/max monthly temperature) plus terrain heights to produce:
//! - land mask (u8 0/255)
//! - biome IDs (u8; 0 reserved for non-land)
//! - derived maps: roughness, albedo suggestion, vegetation density (all f32 in [0,1])

mod config;

pub use config::BiomeConfig;

use glam::Vec3;

use crate::geometry::{face_uv_to_cube, spherify_point, neighbors::neighbor_4, CubeFaceId};
use crate::noise::FractalNoiseConfig;

/// Biome classification ID. `as_u8()` is stable and used for storage/export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BiomeId {
    // 0 is reserved for non-land / masked-out (ocean).
    IceCap = 1,
    Tundra = 2,
    BorealForest = 3,
    TemperateGrassland = 4,
    TemperateDeciduousForest = 5,
    TemperateRainforest = 6,
    SubtropicalDesert = 7,
    Savanna = 8,
    TropicalSeasonalForest = 9,
    TropicalRainforest = 10,
    Mountain = 11,
    Wetlands = 12,
}

impl BiomeId {
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Base albedo suggestion for this biome (0..1).
    pub fn base_albedo(self) -> f32 {
        match self {
            BiomeId::IceCap => 0.78,
            BiomeId::Tundra => 0.30,
            BiomeId::BorealForest => 0.12,
            BiomeId::TemperateGrassland => 0.18,
            BiomeId::TemperateDeciduousForest => 0.14,
            BiomeId::TemperateRainforest => 0.12,
            BiomeId::SubtropicalDesert => 0.40,
            BiomeId::Savanna => 0.20,
            BiomeId::TropicalSeasonalForest => 0.13,
            BiomeId::TropicalRainforest => 0.12,
            BiomeId::Mountain => 0.22,
            BiomeId::Wetlands => 0.16,
        }
    }

    /// Base vegetation density for this biome (0..1).
    pub fn base_veg(self) -> f32 {
        match self {
            BiomeId::IceCap => 0.0,
            BiomeId::Tundra => 0.15,
            BiomeId::BorealForest => 0.65,
            BiomeId::TemperateGrassland => 0.45,
            BiomeId::TemperateDeciduousForest => 0.70,
            BiomeId::TemperateRainforest => 0.85,
            BiomeId::SubtropicalDesert => 0.05,
            BiomeId::Savanna => 0.35,
            BiomeId::TropicalSeasonalForest => 0.75,
            BiomeId::TropicalRainforest => 0.95,
            BiomeId::Mountain => 0.25,
            BiomeId::Wetlands => 0.80,
        }
    }

    /// RGB preview color for this biome.
    pub fn preview_rgb(self) -> [u8; 3] {
        match self {
            BiomeId::IceCap => [240, 248, 255],
            BiomeId::Tundra => [170, 190, 170],
            BiomeId::BorealForest => [25, 85, 45],
            BiomeId::TemperateGrassland => [120, 185, 95],
            BiomeId::TemperateDeciduousForest => [40, 130, 70],
            BiomeId::TemperateRainforest => [15, 110, 75],
            BiomeId::SubtropicalDesert => [228, 210, 145],
            BiomeId::Savanna => [198, 192, 92],
            BiomeId::TropicalSeasonalForest => [55, 165, 85],
            BiomeId::TropicalRainforest => [18, 150, 70],
            BiomeId::Mountain => [140, 140, 140],
            BiomeId::Wetlands => [55, 140, 120],
        }
    }
}

#[derive(Debug, Clone)]
pub struct BiomeOutputs {
    pub land_mask: Vec<u8>,
    pub biome_ids: Vec<u8>,
    pub roughness: Vec<f32>,
    pub albedo: Vec<f32>,
    pub vegetation_density: Vec<f32>,
}

/// Convenience: return an RGB preview color for a biome id (0..=255).
///
/// `0` is reserved for non-land/ocean.
pub fn biome_preview_rgb(biome_id: u8) -> [u8; 3] {
    match biome_id {
        0 => [0, 0, 0],
        1 => BiomeId::IceCap.preview_rgb(),
        2 => BiomeId::Tundra.preview_rgb(),
        3 => BiomeId::BorealForest.preview_rgb(),
        4 => BiomeId::TemperateGrassland.preview_rgb(),
        5 => BiomeId::TemperateDeciduousForest.preview_rgb(),
        6 => BiomeId::TemperateRainforest.preview_rgb(),
        7 => BiomeId::SubtropicalDesert.preview_rgb(),
        8 => BiomeId::Savanna.preview_rgb(),
        9 => BiomeId::TropicalSeasonalForest.preview_rgb(),
        10 => BiomeId::TropicalRainforest.preview_rgb(),
        11 => BiomeId::Mountain.preview_rgb(),
        12 => BiomeId::Wetlands.preview_rgb(),
        _ => [255, 0, 255],
    }
}

/// Classify a single pixel into a biome using a Whittaker-like scheme.
///
/// Inputs should already include ecotone jitter, if enabled.
fn classify_whittaker_like(
    height_km: f32,
    temp_mean_c: f32,
    precip_annual_mm: f32,
    temp_min_month_c: f32,
    roughness_01: f32,
    river_dist_steps: u8,
    cfg: &BiomeConfig,
) -> BiomeId {
    // Mountains get a distinct biome at high elevation; also reduces vegetation later.
    if height_km >= cfg.sea_level + 3.0 {
        return BiomeId::Mountain;
    }

    // Wetlands: prefer explicit river adjacency when available; otherwise use a coastal/flat fallback.
    // Keep this early so it can override forest/grassland in wet floodplains.
    if temp_mean_c > 2.0 && temp_mean_c < 28.0 && temp_min_month_c > -8.0 {
        // Hydrology-proxied trigger:
        // widen floodplains around rivers when very wet, but avoid rugged areas.
        let wet_enough = precip_annual_mm >= 1200.0;
        let max_steps = if precip_annual_mm >= 2000.0 { 2 } else if precip_annual_mm >= 1400.0 { 1 } else { 0 };
        let flat_enough = roughness_01 <= 0.35;
        if wet_enough && flat_enough && river_dist_steps <= max_steps {
            return BiomeId::Wetlands;
        }
        // Fallback: low relief, near sea level, very wet (coastal marsh / deltas)
        let near_sea = height_km <= cfg.sea_level + 0.25;
        let flat = roughness_01 <= 0.12;
        if near_sea && flat && precip_annual_mm >= 1800.0 {
            return BiomeId::Wetlands;
        }
    }

    // Permanent ice: sustained cold.
    if temp_mean_c <= -10.0 || temp_min_month_c <= -25.0 {
        return BiomeId::IceCap;
    }

    // Tundra / boreal transition.
    if temp_mean_c < 0.0 {
        if precip_annual_mm < 350.0 {
            return BiomeId::Tundra;
        }
        return BiomeId::BorealForest;
    }

    // Warm climates.
    if temp_mean_c >= 20.0 {
        if precip_annual_mm >= 2500.0 {
            return BiomeId::TropicalRainforest;
        }
        if precip_annual_mm >= 1200.0 {
            return BiomeId::TropicalSeasonalForest;
        }
        if precip_annual_mm >= 450.0 {
            return BiomeId::Savanna;
        }
        return BiomeId::SubtropicalDesert;
    }

    // Temperate climates.
    if precip_annual_mm >= 1600.0 {
        return BiomeId::TemperateRainforest;
    }
    if precip_annual_mm >= 800.0 {
        return BiomeId::TemperateDeciduousForest;
    }
    if precip_annual_mm >= 350.0 {
        return BiomeId::TemperateGrassland;
    }
    BiomeId::SubtropicalDesert
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn snow_overlay(temp_min_month_c: f32, cfg: &BiomeConfig) -> f32 {
    // 1 at/below freeze, 0 at/above melt.
    if cfg.snow_freeze_temp_c >= cfg.snow_melt_temp_c {
        return if temp_min_month_c <= cfg.snow_freeze_temp_c { 1.0 } else { 0.0 };
    }
    1.0 - smoothstep(cfg.snow_freeze_temp_c, cfg.snow_melt_temp_c, temp_min_month_c)
}

fn jitter_noise_at(
    p: Vec3,
    seed: u64,
    frequency: f32,
) -> f32 {
    // Use the existing fractal noise sampler; single-octave config for cheap jitter.
    let ncfg = FractalNoiseConfig {
        octaves: 1,
        frequency,
        lacunarity: 2.0,
        persistence: 0.5,
        seed: (seed as i32) ^ 0x5a5a_1234,
    };
    crate::noise::sample_fractal_noise(p, &ncfg)
}

fn global_index(res: u32, face: CubeFaceId, x: u32, y: u32) -> usize {
    let per_face = (res * res) as usize;
    face.index() * per_face + (y * res + x) as usize
}

fn for_each_neighbor8<F: FnMut(CubeFaceId, u32, u32)>(
    res: u32,
    face: CubeFaceId,
    x: u32,
    y: u32,
    mut f: F,
) {
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let (mut ff, mut px, mut py) = (face, x, y);
            if dx != 0 {
                (ff, px, py) = neighbor_4(res, ff, px, py, dx, 0);
            }
            if dy != 0 {
                (ff, px, py) = neighbor_4(res, ff, px, py, 0, dy);
            }
            f(ff, px, py);
        }
    }
}

/// Compute biome outputs for a full cube-sphere planet grid.
///
/// All inputs are flattened in face order, length `6*res*res`.
pub fn compute_biomes(
    res: u32,
    cfg: &BiomeConfig,
    heights_km: &[f32],
    temp_mean_c: &[f32],
    precip_annual_mm: &[f32],
    temp_min_month_c: &[f32],
    river_mask: Option<&[u8]>,
) -> BiomeOutputs {
    let per_face = (res * res) as usize;
    let total = per_face * 6;
    assert_eq!(heights_km.len(), total);
    assert_eq!(temp_mean_c.len(), total);
    assert_eq!(precip_annual_mm.len(), total);
    assert_eq!(temp_min_month_c.len(), total);
    if let Some(r) = river_mask {
        assert_eq!(r.len(), total);
    }

    let mut land_mask = vec![0u8; total];
    let mut biome_ids = vec![0u8; total];
    let mut roughness = vec![0.0f32; total];
    let mut albedo = vec![0.0f32; total];
    let mut vegetation_density = vec![0.0f32; total];

    // First pass: roughness (uses seam-aware neighbors).
    // We compute for all pixels (including ocean) but will mask later.
    for face in CubeFaceId::all() {
        for y in 0..res {
            for x in 0..res {
                let i = global_index(res, face, x, y);
                let h0 = heights_km[i];

                let mut sum2 = 0.0f32;
                let mut n = 0.0f32;
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let (f2, x2, y2) = neighbor_4(res, face, x, y, dx, dy);
                    let j = global_index(res, f2, x2, y2);
                    let dh = heights_km[j] - h0;
                    sum2 += dh * dh;
                    n += 1.0;
                }
                let rms = (sum2 / n.max(1.0)).sqrt();
                roughness[i] = (rms * cfg.roughness_scale).clamp(0.0, 1.0);
            }
        }
    }

    // River proximity (0..2 steps), if river mask is provided.
    //
    // Expansion is anisotropic: it prefers low-lying floodplains by limiting uphill steps.
    // This keeps wetlands from climbing aggressively into mountainous terrain while still
    // widening along valleys.
    let (river0, near1, near2): (Option<Vec<u8>>, Option<Vec<u8>>, Option<Vec<u8>>) = if let Some(r) = river_mask {
        // Maximum uphill (km) allowed per expansion step.
        // (kept as a code constant per your preference)
        const MAX_UPHILL_PER_STEP_KM: f32 = 0.04;

        let mut r0 = vec![0u8; total];
        for i in 0..total {
            r0[i] = if r[i] >= 128 { 1 } else { 0 };
        }

        let mut n1 = vec![0u8; total];
        for face in CubeFaceId::all() {
            for y in 0..res {
                for x in 0..res {
                    let i = global_index(res, face, x, y);
                    if r0[i] == 1 {
                        n1[i] = 1;
                        continue;
                    }
                    let mut any = false;
                    for_each_neighbor8(res, face, x, y, |f2, x2, y2| {
                        if any {
                            return;
                        }
                        let j = global_index(res, f2, x2, y2);
                        if r0[j] == 1 && heights_km[i] <= heights_km[j] + MAX_UPHILL_PER_STEP_KM {
                            any = true;
                        }
                    });
                    n1[i] = if any { 1 } else { 0 };
                }
            }
        }

        let mut n2 = vec![0u8; total];
        for face in CubeFaceId::all() {
            for y in 0..res {
                for x in 0..res {
                    let i = global_index(res, face, x, y);
                    if n1[i] == 1 {
                        n2[i] = 1;
                        continue;
                    }
                    let mut any = false;
                    for_each_neighbor8(res, face, x, y, |f2, x2, y2| {
                        if any {
                            return;
                        }
                        let j = global_index(res, f2, x2, y2);
                        if n1[j] == 1 && heights_km[i] <= heights_km[j] + MAX_UPHILL_PER_STEP_KM {
                            any = true;
                        }
                    });
                    n2[i] = if any { 1 } else { 0 };
                }
            }
        }

        (Some(r0), Some(n1), Some(n2))
    } else {
        (None, None, None)
    };

    // Second pass: biome classification + derived maps.
    for face in CubeFaceId::all() {
        for y in 0..res {
            for x in 0..res {
                let i = global_index(res, face, x, y);
                let h = heights_km[i];
                if h <= cfg.sea_level {
                    // ocean: keep masked out
                    continue;
                }
                land_mask[i] = 255;

                let (u, v) = ((x as f32 + 0.5) / res as f32, (y as f32 + 0.5) / res as f32);
                let cube_pos = face_uv_to_cube(face, u, v);
                let sphere_pos = spherify_point(cube_pos);

                let j = jitter_noise_at(sphere_pos, cfg.seed, cfg.jitter_frequency);
                let t = temp_mean_c[i] + cfg.jitter_temp_c * j;
                let p = (precip_annual_mm[i] + cfg.jitter_precip_mm * j).max(0.0);
                let tmin = temp_min_month_c[i];

                let river_dist_steps: u8 = match (&river0, &near1, &near2) {
                    (Some(r0), Some(n1), Some(n2)) => {
                        if r0[i] == 1 { 0 } else if n1[i] == 1 { 1 } else if n2[i] == 1 { 2 } else { u8::MAX }
                    }
                    _ => u8::MAX,
                };

                let biome = classify_whittaker_like(h, t, p, tmin, roughness[i], river_dist_steps, cfg);
                biome_ids[i] = biome.as_u8();

                // Vegetation: combine biome base with climate signal.
                let mut veg = biome.base_veg();
                let temp_w = smoothstep(-5.0, 25.0, t);
                let precip_w = smoothstep(100.0, 2200.0, p);
                veg *= 0.35 + 0.65 * temp_w * precip_w;

                // Mountains reduce vegetation with roughness/elevation.
                if biome == BiomeId::Mountain {
                    veg *= 0.5 * (1.0 - roughness[i]).clamp(0.2, 1.0);
                }

                // Optional river proximity boost.
                if river_dist_steps <= 2 {
                    // Strongest on-river, taper within 2 pixels.
                    let w = 1.0 - (river_dist_steps as f32) / 3.0;
                    veg = (veg + cfg.river_veg_boost * w).clamp(0.0, 1.0);
                }

                vegetation_density[i] = veg.clamp(0.0, 1.0);

                // Albedo: biome base + snow overlay; slightly darken with vegetation.
                let snow = snow_overlay(tmin, cfg);
                let base = biome.base_albedo();
                let mut a = base * (1.0 - 0.25 * vegetation_density[i]);
                a = a * (1.0 - snow) + 0.80 * snow;
                albedo[i] = a.clamp(0.0, 1.0);
            }
        }
    }

    BiomeOutputs {
        land_mask,
        biome_ids,
        roughness,
        albedo,
        vegetation_density,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oceans_are_masked_out() {
        let res = 4;
        let total = (res * res) as usize * 6;
        let cfg = BiomeConfig { sea_level: 0.0, seed: 123, ..Default::default() };

        let heights = vec![0.0f32; total]; // all ocean at sea level
        let t = vec![10.0f32; total];
        let p = vec![1000.0f32; total];
        let tmin = vec![0.0f32; total];

        let out = compute_biomes(res, &cfg, &heights, &t, &p, &tmin, None);
        assert!(out.land_mask.iter().all(|&m| m == 0));
        assert!(out.biome_ids.iter().all(|&b| b == 0));
        assert!(out.roughness.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn jitter_is_deterministic_for_seed() {
        let res = 8;
        let total = (res * res) as usize * 6;
        let heights = vec![0.25f32; total];
        let t = vec![15.0f32; total];
        let p = vec![800.0f32; total];
        let tmin = vec![2.0f32; total];

        let cfg1 = BiomeConfig { sea_level: 0.0, seed: 999, ..Default::default() };
        let cfg2 = BiomeConfig { sea_level: 0.0, seed: 999, ..Default::default() };

        let o1 = compute_biomes(res, &cfg1, &heights, &t, &p, &tmin, None);
        let o2 = compute_biomes(res, &cfg2, &heights, &t, &p, &tmin, None);
        assert_eq!(o1.biome_ids, o2.biome_ids);
    }

    #[test]
    fn derived_maps_are_in_range() {
        let res = 8;
        let total = (res * res) as usize * 6;
        let cfg = BiomeConfig { sea_level: 0.0, seed: 1, ..Default::default() };

        let heights = vec![0.5f32; total];
        let t = vec![25.0f32; total];
        let p = vec![2500.0f32; total];
        let tmin = vec![15.0f32; total];
        let out = compute_biomes(res, &cfg, &heights, &t, &p, &tmin, None);

        assert!(out.roughness.iter().all(|&v| v >= 0.0 && v <= 1.0));
        assert!(out.albedo.iter().all(|&v| v >= 0.0 && v <= 1.0));
        assert!(out.vegetation_density.iter().all(|&v| v >= 0.0 && v <= 1.0));
    }

    #[test]
    fn wetlands_trigger_on_river_when_wet() {
        let res = 8;
        let total = (res * res) as usize * 6;
        let cfg = BiomeConfig { sea_level: 0.0, seed: 0, ..Default::default() };

        // Land everywhere.
        let heights = vec![0.15f32; total];
        // Temperate.
        let t = vec![18.0f32; total];
        // Wet.
        let p = vec![2200.0f32; total];
        let tmin = vec![5.0f32; total];

        // Put a river pixel in the middle of face 0.
        let mut river = vec![0u8; total];
        let mid = (res / 2) as usize;
        let idx = 0 * (res as usize * res as usize) + mid * (res as usize) + mid;
        river[idx] = 255;

        let out = compute_biomes(res, &cfg, &heights, &t, &p, &tmin, Some(&river));
        assert_eq!(out.biome_ids[idx], BiomeId::Wetlands.as_u8());
    }

    #[test]
    fn wetlands_expand_to_adjacent_pixels_when_very_wet() {
        let res = 8;
        let total = (res * res) as usize * 6;
        let cfg = BiomeConfig { sea_level: 0.0, seed: 0, ..Default::default() };

        // Land everywhere, temperate, very wet (enables radius 2).
        let heights = vec![0.15f32; total];
        let t = vec![18.0f32; total];
        let p = vec![2600.0f32; total];
        let tmin = vec![5.0f32; total];

        // River pixel at (mid, mid) on face 0.
        let mut river = vec![0u8; total];
        let per_face = (res as usize) * (res as usize);
        let mid = (res / 2) as usize;
        let center = 0 * per_face + mid * (res as usize) + mid;
        river[center] = 255;

        // Adjacent pixel (mid+1, mid).
        let adj = 0 * per_face + mid * (res as usize) + (mid + 1);

        let out = compute_biomes(res, &cfg, &heights, &t, &p, &tmin, Some(&river));
        assert_eq!(out.biome_ids[center], BiomeId::Wetlands.as_u8());
        assert_eq!(out.biome_ids[adj], BiomeId::Wetlands.as_u8());
    }

    #[test]
    fn wetlands_expansion_resists_uphill() {
        let res = 8;
        let total = (res * res) as usize * 6;
        let cfg = BiomeConfig { sea_level: 0.0, seed: 0, ..Default::default() };

        // Mostly flat land.
        let mut heights = vec![0.15f32; total];
        let t = vec![18.0f32; total];
        let p = vec![2600.0f32; total]; // very wet => allows radius 2 (if reachable)
        let tmin = vec![5.0f32; total];

        let per_face = (res as usize) * (res as usize);
        let mid = (res / 2) as usize;
        let center = 0 * per_face + mid * (res as usize) + mid;

        // Create a small height gradient around the river:
        // - right neighbor is slightly downhill (should allow wetlands)
        // - left neighbor is too uphill (should resist wetlands)
        heights[center] = 0.20;
        let right = 0 * per_face + mid * (res as usize) + (mid + 1);
        let left = 0 * per_face + mid * (res as usize) + (mid - 1);
        heights[right] = 0.19;
        heights[left] = 0.26; // +0.06 km relative to river cell; should be blocked by uphill limit

        let mut river = vec![0u8; total];
        river[center] = 255;

        let out = compute_biomes(res, &cfg, &heights, &t, &p, &tmin, Some(&river));
        assert_eq!(out.biome_ids[center], BiomeId::Wetlands.as_u8());
        assert_eq!(
            out.biome_ids[right],
            BiomeId::Wetlands.as_u8(),
            "downhill neighbor should become wetlands"
        );
        assert_ne!(
            out.biome_ids[left],
            BiomeId::Wetlands.as_u8(),
            "uphill neighbor should resist wetlands expansion"
        );
    }
}

