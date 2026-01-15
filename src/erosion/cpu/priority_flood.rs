//! Seam-aware Priority-Flood depression filling (Barnes et al. 2014).
//!
//! Note: On a closed sphere there is no boundary outlet. We use an `OutletModel`
//! to seed outlets (e.g. sea level).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::erosion::OutletModel;
use crate::geometry::CubeFaceId;
use crate::geometry::neighbors::neighbor_4;

#[derive(Clone, Copy, Debug)]
struct HeapItem {
    height: f32,
    idx: u32,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.height == other.height && self.idx == other.idx
    }
}

impl Eq for HeapItem {}

// Min-heap by height via reversed ordering.
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse for min-heap behavior with BinaryHeap.
        other.height.partial_cmp(&self.height).or(Some(Ordering::Equal))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

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

/// Returns filled heights (same length as input) using Priority-Flood.
pub fn priority_flood_fill(resolution: u32, heights: &[f32], outlet: OutletModel) -> Vec<f32> {
    let per_face = (resolution * resolution) as usize;
    let total = per_face * 6;
    assert_eq!(heights.len(), total);

    let mut filled = heights.to_vec();
    let mut visited = vec![false; total];
    let mut heap = BinaryHeap::<HeapItem>::new();

    // Seed outlets
    match outlet {
        OutletModel::SeaLevel { sea_level } => {
            for idx in 0..total {
                if heights[idx] <= sea_level {
                    visited[idx] = true;
                    heap.push(HeapItem {
                        height: heights[idx],
                        idx: idx as u32,
                    });
                }
            }

            // If nothing is below sea level, fall back to seeding the global minimum.
            if heap.is_empty() {
                let (min_i, min_h) = heights
                    .iter()
                    .enumerate()
                    .fold((0usize, f32::INFINITY), |acc, (i, &h)| {
                        if h < acc.1 { (i, h) } else { acc }
                    });
                visited[min_i] = true;
                heap.push(HeapItem { height: min_h, idx: min_i as u32 });
            }
        }
    }

    while let Some(HeapItem { height: h_cur, idx }) = heap.pop() {
        let (face, x, y) = decode_index(resolution, idx);

        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let (n_face, n_x, n_y) = neighbor_4(resolution, face, x, y, dx, dy);
            let n_idx = global_index(resolution, n_face, n_x, n_y) as usize;
            if visited[n_idx] {
                continue;
            }
            visited[n_idx] = true;

            let h_n = filled[n_idx];
            let new_h = h_n.max(h_cur);
            filled[n_idx] = new_h;
            heap.push(HeapItem { height: new_h, idx: n_idx as u32 });
        }
    }

    filled
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::CubeFaceId;

    #[test]
    fn test_priority_flood_fills_simple_depression_to_rim() {
        let res = 8;
        let per_face = (res * res) as usize;
        let total = per_face * 6;

        // Start with all sea at 0.0.
        let mut h = vec![0.0f32; total];

        // Build a small "basin" on PosZ:
        // - center at 1.0
        // - N/E/S/W rim at 2.0
        let face = CubeFaceId::PosZ;
        let cx = res / 2;
        let cy = res / 2;
        let idx = |x: u32, y: u32| -> usize {
            (face.index() * per_face) + (y as usize * res as usize + x as usize)
        };

        h[idx(cx, cy)] = 1.0;
        h[idx(cx + 1, cy)] = 2.0;
        h[idx(cx - 1, cy)] = 2.0;
        h[idx(cx, cy + 1)] = 2.0;
        h[idx(cx, cy - 1)] = 2.0;

        let filled = priority_flood_fill(res, &h, OutletModel::SeaLevel { sea_level: 0.0 });

        // The center depression should fill to the rim height (2.0).
        assert!((filled[idx(cx, cy)] - 2.0).abs() < 1e-6);
    }
}

