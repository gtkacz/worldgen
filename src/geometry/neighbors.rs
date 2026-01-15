//! Seam-aware cube-face neighbor mapping.
//!
//! This is used by CPU hydrology (Priority-Flood, rivers) and must match the
//! WGSL mapping used by compute shaders.
//!
//! The transforms are derived from the face orientation defined in
//! `face_uv_to_cube` in `cube_sphere.rs`.

use super::CubeFaceId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Edge {
    Left,
    Right,
    Down,
    Up,
}

fn map_edge(resolution: u32, face: CubeFaceId, x: u32, y: u32, edge: Edge) -> (CubeFaceId, u32, u32) {
    debug_assert!(resolution >= 2);
    let r = resolution - 1;

    match (face, edge) {
        // --- PosX ---
        (CubeFaceId::PosX, Edge::Left) => (CubeFaceId::PosZ, r, y),
        (CubeFaceId::PosX, Edge::Right) => (CubeFaceId::NegZ, 0, y),
        (CubeFaceId::PosX, Edge::Down) => (CubeFaceId::NegY, r, x),
        (CubeFaceId::PosX, Edge::Up) => (CubeFaceId::PosY, r, r - x),

        // --- NegX ---
        (CubeFaceId::NegX, Edge::Left) => (CubeFaceId::NegZ, r, y),
        (CubeFaceId::NegX, Edge::Right) => (CubeFaceId::PosZ, 0, y),
        (CubeFaceId::NegX, Edge::Down) => (CubeFaceId::NegY, 0, r - x),
        (CubeFaceId::NegX, Edge::Up) => (CubeFaceId::PosY, 0, x),

        // --- PosY ---
        (CubeFaceId::PosY, Edge::Left) => (CubeFaceId::NegX, y, r),
        (CubeFaceId::PosY, Edge::Right) => (CubeFaceId::PosX, r - y, r),
        (CubeFaceId::PosY, Edge::Down) => (CubeFaceId::NegZ, r - x, r),
        (CubeFaceId::PosY, Edge::Up) => (CubeFaceId::PosZ, x, r),

        // --- NegY ---
        (CubeFaceId::NegY, Edge::Left) => (CubeFaceId::NegX, r - y, 0),
        (CubeFaceId::NegY, Edge::Right) => (CubeFaceId::PosX, y, 0),
        (CubeFaceId::NegY, Edge::Down) => (CubeFaceId::PosZ, x, 0),
        (CubeFaceId::NegY, Edge::Up) => (CubeFaceId::NegZ, r - x, 0),

        // --- PosZ ---
        (CubeFaceId::PosZ, Edge::Left) => (CubeFaceId::NegX, r, y),
        (CubeFaceId::PosZ, Edge::Right) => (CubeFaceId::PosX, 0, y),
        (CubeFaceId::PosZ, Edge::Down) => (CubeFaceId::NegY, x, 0),
        (CubeFaceId::PosZ, Edge::Up) => (CubeFaceId::PosY, x, r),

        // --- NegZ ---
        (CubeFaceId::NegZ, Edge::Left) => (CubeFaceId::PosX, r, y),
        (CubeFaceId::NegZ, Edge::Right) => (CubeFaceId::NegX, 0, y),
        (CubeFaceId::NegZ, Edge::Down) => (CubeFaceId::NegY, r - x, r),
        (CubeFaceId::NegZ, Edge::Up) => (CubeFaceId::PosY, r - x, 0),
    }
}

/// Returns the 4-neighborhood neighbor (von Neumann) for a pixel on a cube-sphere.
///
/// `dx,dy` must be one of: (-1,0), (1,0), (0,-1), (0,1).
pub fn neighbor_4(
    resolution: u32,
    face: CubeFaceId,
    x: u32,
    y: u32,
    dx: i32,
    dy: i32,
) -> (CubeFaceId, u32, u32) {
    debug_assert!(x < resolution && y < resolution);
    debug_assert!(
        (dx.abs() + dy.abs()) == 1,
        "neighbor_4 expects cardinal direction"
    );

    let nx = x as i32 + dx;
    let ny = y as i32 + dy;

    if (0..resolution as i32).contains(&nx) && (0..resolution as i32).contains(&ny) {
        return (face, nx as u32, ny as u32);
    }

    if nx < 0 {
        return map_edge(resolution, face, x, y, Edge::Left);
    }
    if nx >= resolution as i32 {
        return map_edge(resolution, face, x, y, Edge::Right);
    }
    if ny < 0 {
        return map_edge(resolution, face, x, y, Edge::Down);
    }
    debug_assert!(ny >= resolution as i32);
    map_edge(resolution, face, x, y, Edge::Up)
}

/// Returns the 8-neighborhood neighbors (Moore neighborhood) as `(dx, dy, face, x, y)` tuples.
///
/// Diagonals are resolved by applying the x-step (if any) first, then the y-step.
pub fn neighbor_8(
    resolution: u32,
    face: CubeFaceId,
    x: u32,
    y: u32,
) -> Vec<(i32, i32, CubeFaceId, u32, u32)> {
    let mut out = Vec::with_capacity(8);

    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }

            let (mut f, mut px, mut py) = (face, x, y);
            if dx != 0 {
                (f, px, py) = neighbor_4(resolution, f, px, py, dx, 0);
            }
            if dy != 0 {
                (f, px, py) = neighbor_4(resolution, f, px, py, 0, dy);
            }
            out.push((dx, dy, f, px, py));
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::FaceCoord;

    #[test]
    fn test_neighbor_4_bidirectional_adjacency() {
        let res = 16;
        let r = res - 1;

        for face in CubeFaceId::all() {
            // Sample edge cells only (most likely to be wrong).
            let samples = [
                (0, 0),
                (0, r),
                (r, 0),
                (r, r),
                (0, r / 2),
                (r, r / 2),
                (r / 2, 0),
                (r / 2, r),
            ];

            for &(x, y) in &samples {
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let (f2, x2, y2) = neighbor_4(res, face, x, y, dx, dy);

                    // The reverse direction may not be the negated (dx,dy) due to face rotations.
                    // Instead, ensure the original cell appears in the neighbor set of the mapped cell.
                    let back_neighbors: Vec<(CubeFaceId, u32, u32)> = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                        .into_iter()
                        .map(|(bdx, bdy)| neighbor_4(res, f2, x2, y2, bdx, bdy))
                        .collect();

                    assert!(
                        back_neighbors.contains(&(face, x, y)),
                        "Adjacency not bidirectional: {:?} ({},{}) -> {:?} ({},{}); back={:?}",
                        face,
                        x,
                        y,
                        f2,
                        x2,
                        y2,
                        back_neighbors
                    );
                }
            }
        }
    }

    #[test]
    fn test_neighbor_geometric_continuity() {
        // Adjacent neighbors should be close on the unit sphere.
        let res = 64;
        let r = res - 1;

        let samples = [
            (0, 0),
            (0, r),
            (r, 0),
            (r, r),
            (0, r / 2),
            (r, r / 2),
            (r / 2, 0),
            (r / 2, r),
        ];

        for face in CubeFaceId::all() {
            for &(x, y) in &samples {
                let u = (x as f32 + 0.5) / res as f32;
                let v = (y as f32 + 0.5) / res as f32;
                let p0 = FaceCoord::new(face, u, v).to_sphere_point();

                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let (f2, x2, y2) = neighbor_4(res, face, x, y, dx, dy);
                    let u2 = (x2 as f32 + 0.5) / res as f32;
                    let v2 = (y2 as f32 + 0.5) / res as f32;
                    let p1 = FaceCoord::new(f2, u2, v2).to_sphere_point();

                    let dot = p0.dot(p1).clamp(-1.0, 1.0);
                    let angle = dot.acos();
                    assert!(
                        angle < 0.2,
                        "Neighbor too far: face {:?} ({},{}) -> {:?} ({},{}), angle={}",
                        face,
                        x,
                        y,
                        f2,
                        x2,
                        y2,
                        angle
                    );
                }
            }
        }
    }
}

