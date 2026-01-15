//! CPU hydraulic + thermal erosion backend (Phase 3 fallback).
//!
//! This is designed to mirror the GPU WGSL pipeline in `erosion/wgpu/shaders/hydraulic.wgsl`
//! while remaining deterministic and seam-aware on the cube-sphere.

use crate::erosion::ErosionConfig;
use crate::geometry::neighbors::neighbor_4;
use crate::geometry::CubeFaceId;

/// Outputs produced by the CPU erosion backend.
#[derive(Debug, Clone)]
pub struct ErosionCpuOutputs {
    pub heights: Vec<f32>,
    pub water: Vec<f32>,
    pub sediment: Vec<f32>,
    /// Net height delta caused by hydraulic+thermal erosion (excluding depression filling).
    pub deposition: Vec<f32>,
}

fn global_index(res: u32, face: CubeFaceId, x: u32, y: u32) -> usize {
    let per_face = (res * res) as usize;
    (face.index() * per_face) + (y as usize * res as usize + x as usize)
}

fn decode_index(res: u32, idx: usize) -> (CubeFaceId, u32, u32) {
    let per_face = (res * res) as usize;
    let face_i = idx / per_face;
    let local = idx % per_face;
    let x = (local % res as usize) as u32;
    let y = (local / res as usize) as u32;
    (CubeFaceId::from_index(face_i).unwrap(), x, y)
}

#[inline]
fn clamp01(x: f32) -> f32 {
    x.max(0.0).min(1.0)
}

/// Run CPU hydraulic erosion for `config.hydraulic_steps`, then mass-conserving thermal erosion for
/// `config.thermal_iterations`.
///
/// Input heights must be a flat array of length `resolution*resolution*6` in face-major order.
pub fn run_hydraulic_thermal_cpu(resolution: u32, heights0: &[f32], config: &ErosionConfig) -> ErosionCpuOutputs {
    let per_face = (resolution * resolution) as usize;
    let total = per_face * 6;
    assert_eq!(heights0.len(), total);
    assert!(resolution >= 2);

    let mut h = heights0.to_vec();
    let mut w = vec![0.0f32; total];
    let mut s = vec![0.0f32; total];

    // flux = (E, W, N, S)
    let mut flux = vec![[0.0f32; 4]; total];

    // net delta (height_final - height_initial) for hydraulic+thermal, excluding depression filling
    let mut deposition = vec![0.0f32; total];

    let mut w_next = vec![0.0f32; total];
    let mut h_next = vec![0.0f32; total];
    let mut s_next = vec![0.0f32; total];
    let mut flux_next = vec![[0.0f32; 4]; total];

    let rainfall = config.rainfall;
    let evap = clamp01(config.evaporation);
    let ks = config.erosion_rate.max(0.0);
    let kd = config.deposition_rate.max(0.0);
    let kc = config.sediment_capacity.max(0.0);

    for _step in 0..config.hydraulic_steps {
        // Pass 1 (rainfall): in-place for CPU.
        if rainfall != 0.0 {
            for wi in &mut w {
                *wi += rainfall;
            }
        }

        // Pass 2 (flow): compute outflow fluxes.
        for i in 0..total {
            let (face, x, y) = decode_index(resolution, i);
            let total_hw = h[i] + w[i];

            let (fe, ex, ey) = {
                let (f2, x2, y2) = neighbor_4(resolution, face, x, y, 1, 0);
                (f2, x2, y2)
            };
            let (fw, wx, wy) = {
                let (f2, x2, y2) = neighbor_4(resolution, face, x, y, -1, 0);
                (f2, x2, y2)
            };
            let (fn_, nx, ny) = {
                let (f2, x2, y2) = neighbor_4(resolution, face, x, y, 0, 1);
                (f2, x2, y2)
            };
            let (fs_, sx, sy) = {
                let (f2, x2, y2) = neighbor_4(resolution, face, x, y, 0, -1);
                (f2, x2, y2)
            };

            let te_i = global_index(resolution, fe, ex, ey);
            let tw_i = global_index(resolution, fw, wx, wy);
            let tn_i = global_index(resolution, fn_, nx, ny);
            let ts_i = global_index(resolution, fs_, sx, sy);

            let te = h[te_i] + w[te_i];
            let tw = h[tw_i] + w[tw_i];
            let tn = h[tn_i] + w[tn_i];
            let ts = h[ts_i] + w[ts_i];

            let mut f_e = (total_hw - te).max(0.0);
            let mut f_w = (total_hw - tw).max(0.0);
            let mut f_n = (total_hw - tn).max(0.0);
            let mut f_s = (total_hw - ts).max(0.0);

            let sum = f_e + f_w + f_n + f_s;
            if sum > 1e-6 && sum > w[i] {
                let scale = w[i] / sum;
                f_e *= scale;
                f_w *= scale;
                f_n *= scale;
                f_s *= scale;
            }

            flux_next[i] = [f_e, f_w, f_n, f_s];
        }
        std::mem::swap(&mut flux, &mut flux_next);

        // Pass 3 (water update): apply flux divergence + evaporation.
        for i in 0..total {
            let (face, x, y) = decode_index(resolution, i);
            let f = flux[i];
            let outflow = f[0] + f[1] + f[2] + f[3];

            let (w_face, w_x, w_y) = neighbor_4(resolution, face, x, y, -1, 0);
            let (e_face, e_x, e_y) = neighbor_4(resolution, face, x, y, 1, 0);
            let (s_face, s_x, s_y) = neighbor_4(resolution, face, x, y, 0, -1);
            let (n_face, n_x, n_y) = neighbor_4(resolution, face, x, y, 0, 1);

            let west_i = global_index(resolution, w_face, w_x, w_y);
            let east_i = global_index(resolution, e_face, e_x, e_y);
            let south_i = global_index(resolution, s_face, s_x, s_y);
            let north_i = global_index(resolution, n_face, n_x, n_y);

            // Inflow from neighbors: west's E, east's W, south's N, north's S.
            let inflow = flux[west_i][0] + flux[east_i][1] + flux[south_i][2] + flux[north_i][3];

            let w1 = (w[i] + inflow - outflow).max(0.0);
            w_next[i] = w1 * (1.0 - evap);
        }
        std::mem::swap(&mut w, &mut w_next);

        // Pass 4 (erosion/deposition): update h/s.
        for i in 0..total {
            let (face, x, y) = decode_index(resolution, i);
            let total_hw = h[i] + w[i];

            let (fe, ex, ey) = neighbor_4(resolution, face, x, y, 1, 0);
            let (fw, wx, wy) = neighbor_4(resolution, face, x, y, -1, 0);
            let (fn_, nx, ny) = neighbor_4(resolution, face, x, y, 0, 1);
            let (fs_, sx, sy) = neighbor_4(resolution, face, x, y, 0, -1);

            let te_i = global_index(resolution, fe, ex, ey);
            let tw_i = global_index(resolution, fw, wx, wy);
            let tn_i = global_index(resolution, fn_, nx, ny);
            let ts_i = global_index(resolution, fs_, sx, sy);

            let te = h[te_i] + w[te_i];
            let tw = h[tw_i] + w[tw_i];
            let tn = h[tn_i] + w[tn_i];
            let ts = h[ts_i] + w[ts_i];

            let min_n = te.min(tw).min(tn.min(ts));
            let slope = (total_hw - min_n).max(0.0);

            let f = flux[i];
            let vx = f[0] - f[1];
            let vy = f[2] - f[3];
            let vel = (vx * vx + vy * vy).sqrt();

            let capacity = kc * vel * slope;
            let h0 = h[i];
            let s0 = s[i];

            let (h1, s1) = if s0 > capacity {
                let dep = kd * (s0 - capacity);
                (h0 + dep, s0 - dep)
            } else {
                let ero = ks * (capacity - s0);
                (h0 - ero, s0 + ero)
            };

            deposition[i] += h1 - h0;
            h_next[i] = h1;
            s_next[i] = s1;
        }
        std::mem::swap(&mut h, &mut h_next);
        std::mem::swap(&mut s, &mut s_next);

        // Pass 5 (sediment transport): simple gather from one upstream neighbor.
        for i in 0..total {
            let (face, x, y) = decode_index(resolution, i);
            let f = flux[i];
            let vx = f[0] - f[1];
            let vy = f[2] - f[3];

            let mut src_face = face;
            let mut src_x = x;
            let mut src_y = y;

            if vx.abs() > vy.abs() {
                // Flow to +x means upstream is -x (and vice versa).
                if vx > 0.0 {
                    (src_face, src_x, src_y) = neighbor_4(resolution, face, x, y, -1, 0);
                } else if vx < 0.0 {
                    (src_face, src_x, src_y) = neighbor_4(resolution, face, x, y, 1, 0);
                }
            } else {
                if vy > 0.0 {
                    (src_face, src_x, src_y) = neighbor_4(resolution, face, x, y, 0, -1);
                } else if vy < 0.0 {
                    (src_face, src_x, src_y) = neighbor_4(resolution, face, x, y, 0, 1);
                }
            }

            let src_i = global_index(resolution, src_face, src_x, src_y);
            s_next[i] = s[src_i];
        }
        std::mem::swap(&mut s, &mut s_next);

        // Pass 6 (extra evaporation): mirror WGSL's optional evaporation pass.
        if evap != 0.0 {
            let factor = 1.0 - 0.25 * evap;
            for wi in &mut w {
                *wi = (*wi * factor).max(0.0);
            }
        }
    }

    // Thermal erosion (mass-conserving): redistribute excess above talus to lower neighbors.
    if config.thermal_iterations > 0 {
        run_thermal_cpu(resolution, &mut h, &mut deposition, config);
    }

    if !config.track_deposition {
        deposition.fill(0.0);
    }

    ErosionCpuOutputs {
        heights: h,
        water: w,
        sediment: s,
        deposition,
    }
}

fn run_thermal_cpu(resolution: u32, heights: &mut [f32], deposition: &mut [f32], config: &ErosionConfig) {
    let per_face = (resolution * resolution) as usize;
    let total = per_face * 6;
    assert_eq!(heights.len(), total);
    assert_eq!(deposition.len(), total);

    // Talus threshold: tan(angle) * cell_size.
    let cell_size = 1.0 / resolution as f32;
    let talus = config.angle_of_repose_rad.tan().max(0.0) * cell_size;
    let strength = clamp01(config.thermal_strength);

    let mut delta = vec![0.0f32; total];

    for _ in 0..config.thermal_iterations {
        delta.fill(0.0);

        for i in 0..total {
            let (face, x, y) = decode_index(resolution, i);
            let h0 = heights[i];

            let nbs = [
                neighbor_4(resolution, face, x, y, 1, 0),
                neighbor_4(resolution, face, x, y, -1, 0),
                neighbor_4(resolution, face, x, y, 0, 1),
                neighbor_4(resolution, face, x, y, 0, -1),
            ];

            // Collect lower neighbors with slope above talus.
            let mut candidates: [(usize, f32); 4] = [(0, 0.0); 4];
            let mut count = 0usize;
            for (j, (nf, nx, ny)) in nbs.iter().copied().enumerate() {
                let ni = global_index(resolution, nf, nx, ny);
                let diff = h0 - heights[ni];
                if diff > talus {
                    candidates[count] = (ni, diff - talus);
                    count += 1;
                }
                let _ = j;
            }
            if count == 0 {
                continue;
            }

            // Move a fraction of the excess. Split equally among steep directions.
            let total_excess: f32 = candidates[..count].iter().map(|&(_, ex)| ex).sum();
            let move_amount = strength * total_excess;
            if move_amount <= 0.0 {
                continue;
            }

            let share = move_amount / (count as f32);
            delta[i] -= move_amount;
            for &(ni, _) in &candidates[..count] {
                delta[ni] += share;
            }
        }

        for i in 0..total {
            heights[i] += delta[i];
            deposition[i] += delta[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::erosion::{ErosionBackend, OutletModel};

    #[test]
    fn cpu_erosion_water_non_negative() {
        let res = 16;
        let total = (res * res) as usize * 6;
        let heights0 = vec![0.0f32; total];
        let cfg = ErosionConfig {
            backend: ErosionBackend::CpuOnly,
            hydraulic_steps: 10,
            rainfall: 0.01,
            evaporation: 0.02,
            erosion_rate: 0.02,
            deposition_rate: 0.02,
            sediment_capacity: 0.05,
            thermal_iterations: 0,
            angle_of_repose_rad: 35.0f32.to_radians(),
            thermal_strength: 0.25,
            outlet_model: OutletModel::SeaLevel { sea_level: 0.0 },
            river_accum_threshold: 10,
            keep_intermediates: false,
            track_deposition: true,
        };

        let out = run_hydraulic_thermal_cpu(res, &heights0, &cfg);
        assert!(out.water.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn cpu_deposition_matches_height_delta_sum() {
        let res = 16;
        let total = (res * res) as usize * 6;
        // Simple sloped field: increasing with global index.
        let heights0: Vec<f32> = (0..total).map(|i| i as f32 * 1e-6).collect();
        let cfg = ErosionConfig {
            backend: ErosionBackend::CpuOnly,
            hydraulic_steps: 20,
            rainfall: 0.01,
            evaporation: 0.01,
            erosion_rate: 0.02,
            deposition_rate: 0.02,
            sediment_capacity: 0.05,
            thermal_iterations: 5,
            angle_of_repose_rad: 35.0f32.to_radians(),
            thermal_strength: 0.25,
            outlet_model: OutletModel::SeaLevel { sea_level: 0.0 },
            river_accum_threshold: 10,
            keep_intermediates: false,
            track_deposition: true,
        };

        let out = run_hydraulic_thermal_cpu(res, &heights0, &cfg);
        let sum0: f32 = heights0.iter().sum();
        let sum1: f32 = out.heights.iter().sum();
        let dep: f32 = out.deposition.iter().sum();
        assert!(((sum1 - sum0) - dep).abs() < 1e-3);
    }
}

