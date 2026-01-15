//! River extraction from a heightfield (flow direction + accumulation).

use crate::geometry::CubeFaceId;
use crate::geometry::neighbors::neighbor_8;
use crate::terrain::Planet;

fn global_index(res: u32, face: CubeFaceId, x: u32, y: u32) -> u32 {
    let per_face = res * res;
    face.index() as u32 * per_face + (y * res + x)
}

fn decode_index(res: u32, idx: u32) -> (CubeFaceId, u32, u32) {
    let per_face = res * res;
    let face_i = (idx / per_face) as usize;
    let local = idx % per_face;
    let x = local % res;
    let y = local / res;
    (CubeFaceId::from_index(face_i).unwrap(), x, y)
}

/// For each cell, store downstream cell index (global) or `u32::MAX` if sink/outlet.
pub fn compute_flow_directions_8(resolution: u32, heights: &[f32]) -> Vec<u32> {
    let total = (resolution as usize) * (resolution as usize) * 6;
    assert_eq!(heights.len(), total);

    let mut down = vec![u32::MAX; total];

    for i in 0..total {
        let (face, x, y) = decode_index(resolution, i as u32);
        let h0 = heights[i];

        let mut best_idx = u32::MAX;
        let mut best_h = h0;

        for (dx, dy, n_face, n_x, n_y) in neighbor_8(resolution, face, x, y) {
            let n_i = global_index(resolution, n_face, n_x, n_y) as usize;
            let hn = heights[n_i];
            // Steepest descent: pick strictly lower neighbor; ties ignored.
            if hn < best_h {
                best_h = hn;
                best_idx = n_i as u32;
            }
            // Keep dx/dy used to avoid unused warning if we ever change neighbor_8 to include metadata.
            let _ = (dx, dy);
        }

        down[i] = best_idx;
    }

    down
}

/// Compute contributing area / flow accumulation (cells count), given downstream indices.
pub fn compute_flow_accumulation(resolution: u32, heights: &[f32], downstream: &[u32]) -> Vec<u32> {
    let total = (resolution as usize) * (resolution as usize) * 6;
    assert_eq!(heights.len(), total);
    assert_eq!(downstream.len(), total);

    // Sort indices by height descending so upstream contributions are processed before downstream.
    let mut order: Vec<u32> = (0..total as u32).collect();
    order.sort_by(|&a, &b| {
        let ha = heights[a as usize];
        let hb = heights[b as usize];
        hb.partial_cmp(&ha).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut accum = vec![1u32; total]; // each cell contributes at least 1

    for &i in &order {
        let d = downstream[i as usize];
        if d != u32::MAX {
            let d_usize = d as usize;
            accum[d_usize] = accum[d_usize].saturating_add(accum[i as usize]);
        }
    }

    // Optional: smooth sinks by letting them keep their accumulation only.
    accum
}

/// Create a binary river mask (0/255) from accumulation.
pub fn river_mask_from_accum(accum: &[u32], threshold: u32) -> Vec<u8> {
    accum
        .iter()
        .map(|&a| if a >= threshold { 255 } else { 0 })
        .collect()
}

/// Convenience: compute everything from heights.
pub fn rivers_from_heights(
    resolution: u32,
    heights: &[f32],
    river_threshold: u32,
) -> (Vec<u32>, Vec<u32>, Vec<u8>) {
    let downstream = compute_flow_directions_8(resolution, heights);
    let accum = compute_flow_accumulation(resolution, heights, &downstream);
    let mask = river_mask_from_accum(&accum, river_threshold);
    (downstream, accum, mask)
}

/// Scatter global accumulation + mask arrays into `planet.faces[*].flow_accum` / `river_mask`.
pub fn write_river_outputs_to_planet(planet: &mut Planet, accum: &[u32], mask: &[u8]) {
    let res = planet.resolution();
    let per_face = (res * res) as usize;
    assert_eq!(accum.len(), per_face * 6);
    assert_eq!(mask.len(), per_face * 6);

    for (i, face) in planet.faces.iter_mut().enumerate() {
        let a = &accum[i * per_face..(i + 1) * per_face];
        let m = &mask[i * per_face..(i + 1) * per_face];
        face.flow_accum = Some(a.to_vec());
        face.river_mask = Some(m.to_vec());
    }
}

