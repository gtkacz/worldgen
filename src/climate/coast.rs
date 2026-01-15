//! Coastline distance computation on a cube-sphere grid.

use std::collections::VecDeque;

use crate::geometry::{neighbors::neighbor_4, CubeFaceId};
use crate::terrain::Planet;

/// Compute coastline distance (km) for every pixel.
///
/// Returns a flat array of length `6*res*res` in face order (matching `Planet.faces` order).
/// Ocean pixels have distance 0. Land pixels have distance to the nearest coastline.
pub fn compute_coast_distance_km(planet: &Planet, sea_level: f32) -> Vec<f32> {
    let res = planet.resolution();
    let per_face = (res * res) as usize;
    let total = per_face * 6;

    // Ocean/land mask from current heights.
    let mut is_ocean = vec![false; total];
    for (fi, face) in planet.faces.iter().enumerate() {
        let base = fi * per_face;
        for (i, &h) in face.heights.iter().enumerate() {
            is_ocean[base + i] = h <= sea_level;
        }
    }

    // Identify coastline seeds: land pixels adjacent to ocean.
    let mut dist_steps = vec![u32::MAX; total];
    let mut q: VecDeque<usize> = VecDeque::new();

    for face_id in CubeFaceId::all() {
        let fi = face_id.index();
        let base = fi * per_face;
        for y in 0..res {
            for x in 0..res {
                let idx = base + (y * res + x) as usize;
                if is_ocean[idx] {
                    // Ocean pixels keep distance 0 but are not BFS sources; coast is defined from land side.
                    continue;
                }

                // Land cell: if any cardinal neighbor is ocean -> coastline seed.
                let mut is_coast = false;
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let (nf, nx, ny) = neighbor_4(res, face_id, x, y, dx, dy);
                    let nidx = nf.index() * per_face + (ny * res + nx) as usize;
                    if is_ocean[nidx] {
                        is_coast = true;
                        break;
                    }
                }

                if is_coast {
                    dist_steps[idx] = 0;
                    q.push_back(idx);
                }
            }
        }
    }

    // Multi-source BFS on the full grid to compute shortest step-count distance.
    while let Some(idx) = q.pop_front() {
        let d = dist_steps[idx];
        // Decode (face, x, y)
        let fi = idx / per_face;
        let within = idx - fi * per_face;
        let x = (within as u32) % res;
        let y = (within as u32) / res;
        let face_id = CubeFaceId::from_index(fi).expect("face index 0..6");

        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let (nf, nx, ny) = neighbor_4(res, face_id, x, y, dx, dy);
            let nidx = nf.index() * per_face + (ny * res + nx) as usize;
            if dist_steps[nidx] == u32::MAX {
                dist_steps[nidx] = d.saturating_add(1);
                q.push_back(nidx);
            }
        }
    }

    // Convert step count to km using average arc length per pixel.
    // Each cube face spans 90° (π/2), and has `res` samples across.
    let step_km = (std::f32::consts::FRAC_PI_2 * planet.radius) / res.max(1) as f32;

    let mut out = vec![0.0f32; total];
    for i in 0..total {
        if is_ocean[i] {
            out[i] = 0.0;
        } else {
            let ds = dist_steps[i];
            out[i] = if ds == u32::MAX {
                // No coastline (e.g., all-land planet). Treat as “far inland”.
                step_km * (res as f32)
            } else {
                step_km * (ds as f32)
            };
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coast_distance_is_zero_on_coastline() {
        // Synthetic planet: one land pixel surrounded by ocean.
        let mut planet = Planet::new(8, 1, 6371.0);
        // Make everything ocean
        for face in &mut planet.faces {
            for h in &mut face.heights {
                *h = -1.0;
            }
        }
        // Place a single land cell at PosZ center
        let face = &mut planet.faces[CubeFaceId::PosZ.index()];
        let x = 4;
        let y = 4;
        face.heights[(y * 8 + x) as usize] = 1.0;

        let coast = compute_coast_distance_km(&planet, 0.0);
        let per_face = 8 * 8;
        let idx = CubeFaceId::PosZ.index() * per_face + (y * 8 + x) as usize;
        assert_eq!(coast[idx], 0.0);
    }
}

