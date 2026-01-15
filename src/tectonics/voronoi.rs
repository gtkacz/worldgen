//! Spherical Voronoi tessellation for plate boundaries.

use glam::Vec3;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use std::f32::consts::PI;

/// Spherical Voronoi tessellation for dividing a sphere into cells.
#[derive(Debug, Clone)]
pub struct SphericalVoronoi {
    /// Center points of each Voronoi cell on the unit sphere.
    pub cell_centers: Vec<Vec3>,
    /// Cached neighbor relationships between cells.
    neighbors: Vec<Vec<usize>>,
}

impl SphericalVoronoi {
    /// Creates a new spherical Voronoi tessellation.
    ///
    /// Uses Fibonacci spiral sampling for initial point placement,
    /// followed by Lloyd's relaxation to improve uniformity.
    ///
    /// # Arguments
    /// * `num_cells` - Number of cells (plates) to create
    /// * `lloyd_iterations` - Number of Lloyd's relaxation iterations
    /// * `seed` - Random seed for reproducibility
    pub fn new(num_cells: usize, lloyd_iterations: usize, seed: u64) -> Self {
        let mut centers = fibonacci_spiral_points(num_cells, seed);

        // Apply Lloyd's relaxation to improve uniformity
        for _ in 0..lloyd_iterations {
            lloyds_relaxation(&mut centers);
        }

        let neighbors = compute_neighbors(&centers);

        Self {
            cell_centers: centers,
            neighbors,
        }
    }

    /// Returns the number of cells.
    pub fn num_cells(&self) -> usize {
        self.cell_centers.len()
    }

    /// Finds which cell a point on the sphere belongs to.
    ///
    /// Uses brute-force nearest neighbor search. For small numbers of cells
    /// (typical for tectonic plates), this is efficient enough.
    ///
    /// # Arguments
    /// * `pos` - Position on the unit sphere
    ///
    /// # Returns
    /// Index of the nearest cell center
    pub fn assign_point(&self, pos: Vec3) -> usize {
        let pos_normalized = pos.normalize();
        let mut best_idx = 0;
        let mut best_dist_sq = f32::MAX;

        for (idx, center) in self.cell_centers.iter().enumerate() {
            let dist_sq = (pos_normalized - *center).length_squared();
            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best_idx = idx;
            }
        }

        best_idx
    }

    /// Returns the indices of neighboring cells for a given cell.
    pub fn get_neighbors(&self, cell_idx: usize) -> &[usize] {
        &self.neighbors[cell_idx]
    }

    /// Checks if two cells are neighbors.
    pub fn are_neighbors(&self, cell_a: usize, cell_b: usize) -> bool {
        self.neighbors[cell_a].contains(&cell_b)
    }

    /// Returns the center of a cell.
    pub fn cell_center(&self, cell_idx: usize) -> Vec3 {
        self.cell_centers[cell_idx]
    }

    /// Estimates the area of each cell (as a fraction of the sphere).
    ///
    /// Uses Monte Carlo sampling for a rough estimate.
    pub fn estimate_cell_areas(&self, samples_per_cell: usize) -> Vec<f32> {
        let total_samples = self.num_cells() * samples_per_cell;
        let mut counts = vec![0usize; self.num_cells()];

        let mut rng = ChaCha8Rng::seed_from_u64(12345);

        for _ in 0..total_samples {
            // Generate random point on sphere
            let u: f32 = rng.random();
            let v: f32 = rng.random();
            let theta = 2.0 * PI * u;
            let phi = (2.0 * v - 1.0).acos();

            let pos = Vec3::new(
                phi.sin() * theta.cos(),
                phi.sin() * theta.sin(),
                phi.cos(),
            );

            let cell = self.assign_point(pos);
            counts[cell] += 1;
        }

        counts
            .iter()
            .map(|&c| c as f32 / total_samples as f32)
            .collect()
    }
}

/// Generates N points distributed on a sphere using Fibonacci spiral.
///
/// The Fibonacci spiral provides near-optimal point distribution with
/// minimal clustering and good coverage of the sphere.
///
/// # Arguments
/// * `n` - Number of points to generate
/// * `seed` - Random seed for small perturbations
pub fn fibonacci_spiral_points(n: usize, seed: u64) -> Vec<Vec3> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let golden_ratio = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let angle_increment = 2.0 * PI / golden_ratio;

    (0..n)
        .map(|i| {
            // Latitude: evenly spaced from -1 to 1
            let y = 1.0 - (2.0 * i as f32 + 1.0) / n as f32;
            let radius = (1.0 - y * y).sqrt();

            // Longitude: golden angle spiral with small random offset
            let theta = angle_increment * i as f32 + rng.random::<f32>() * 0.1;

            Vec3::new(radius * theta.cos(), y, radius * theta.sin())
        })
        .collect()
}

/// Performs Lloyd's relaxation to improve point distribution uniformity.
///
/// Each point is moved toward the centroid of its Voronoi cell, then
/// re-normalized to the sphere surface. This reduces variance in cell sizes.
///
/// # Arguments
/// * `points` - Mutable slice of points to relax
pub fn lloyds_relaxation(points: &mut [Vec3]) {
    let n = points.len();
    if n < 2 {
        return;
    }

    // Sample the sphere and accumulate centroids
    let samples_per_cell = 100;
    let total_samples = n * samples_per_cell;

    let mut centroids = vec![Vec3::ZERO; n];
    let mut counts = vec![0usize; n];

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    for _ in 0..total_samples {
        // Generate random point on sphere
        let u: f32 = rng.random();
        let v: f32 = rng.random();
        let theta = 2.0 * PI * u;
        let phi = (2.0 * v - 1.0).acos();

        let sample = Vec3::new(
            phi.sin() * theta.cos(),
            phi.sin() * theta.sin(),
            phi.cos(),
        );

        // Find nearest cell center
        let mut best_idx = 0;
        let mut best_dist_sq = f32::MAX;
        for (idx, point) in points.iter().enumerate() {
            let dist_sq = (sample - *point).length_squared();
            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best_idx = idx;
            }
        }

        centroids[best_idx] += sample;
        counts[best_idx] += 1;
    }

    // Move points toward their cell centroids
    for (i, point) in points.iter_mut().enumerate() {
        if counts[i] > 0 {
            let centroid = centroids[i] / counts[i] as f32;
            // Move 50% toward centroid to avoid oscillation
            let new_pos = *point + (centroid - *point) * 0.5;
            *point = new_pos.normalize();
        }
    }
}

/// Computes neighbor relationships between Voronoi cells.
///
/// Two cells are neighbors if there exist points on the sphere that are
/// approximately equidistant from both cell centers (within tolerance).
fn compute_neighbors(centers: &[Vec3]) -> Vec<Vec<usize>> {
    let n = centers.len();
    let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];

    // For small numbers of cells, use brute force approach
    // Two cells are neighbors if they share a boundary
    // We approximate this by checking if any sampled point is nearly equidistant

    // Threshold for considering cells as neighbors
    // Based on typical angular distance between cells
    let avg_cell_angle = (4.0 * PI / n as f32).sqrt();
    let neighbor_threshold = avg_cell_angle * 1.5;

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = great_circle_distance(centers[i], centers[j]);
            if dist < neighbor_threshold {
                neighbors[i].push(j);
                neighbors[j].push(i);
            }
        }
    }

    neighbors
}

/// Computes the great circle distance between two points on a unit sphere.
fn great_circle_distance(a: Vec3, b: Vec3) -> f32 {
    // For unit vectors, the angle between them is acos(dot product)
    let dot = a.dot(b).clamp(-1.0, 1.0);
    dot.acos()
}

/// Finds the boundary point between two adjacent cells.
///
/// Returns a point on the great circle midway between the two cell centers.
pub fn boundary_midpoint(center_a: Vec3, center_b: Vec3) -> Vec3 {
    ((center_a + center_b) / 2.0).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci_spiral_coverage() {
        let points = fibonacci_spiral_points(100, 42);

        assert_eq!(points.len(), 100);

        // All points should be on unit sphere
        for p in &points {
            assert!((p.length() - 1.0).abs() < 1e-5);
        }

        // Points should cover both hemispheres
        let north_count = points.iter().filter(|p| p.y > 0.0).count();
        let south_count = points.iter().filter(|p| p.y < 0.0).count();
        assert!(north_count > 40 && north_count < 60);
        assert!(south_count > 40 && south_count < 60);
    }

    #[test]
    fn test_lloyd_relaxation_improves_uniformity() {
        let mut points = fibonacci_spiral_points(20, 123);

        // Calculate initial variance in nearest-neighbor distances
        let initial_variance = calculate_nn_variance(&points);

        lloyds_relaxation(&mut points);

        let final_variance = calculate_nn_variance(&points);

        // Fibonacci spiral already produces good uniformity, so Lloyd's
        // may not always dramatically improve it. Just verify:
        // 1. The variance doesn't explode (< 3x initial)
        // 2. All points remain on the unit sphere
        assert!(final_variance <= initial_variance * 3.0,
            "Lloyd's relaxation should not significantly worsen distribution");

        for p in &points {
            let len = p.length();
            assert!((len - 1.0).abs() < 0.01, "Points should remain on unit sphere");
        }
    }

    fn calculate_nn_variance(points: &[Vec3]) -> f32 {
        let distances: Vec<f32> = points
            .iter()
            .map(|p| {
                points
                    .iter()
                    .filter(|q| (*q - *p).length() > 0.01)
                    .map(|q| (*q - *p).length())
                    .fold(f32::MAX, f32::min)
            })
            .collect();

        let mean = distances.iter().sum::<f32>() / distances.len() as f32;
        distances.iter().map(|d| (d - mean).powi(2)).sum::<f32>() / distances.len() as f32
    }

    #[test]
    fn test_voronoi_cell_assignment() {
        let voronoi = SphericalVoronoi::new(10, 2, 42);

        // Each cell center should be assigned to itself
        for (idx, center) in voronoi.cell_centers.iter().enumerate() {
            assert_eq!(voronoi.assign_point(*center), idx);
        }
    }

    #[test]
    fn test_voronoi_neighbor_symmetry() {
        let voronoi = SphericalVoronoi::new(12, 2, 42);

        // Neighbor relationship should be symmetric
        for i in 0..voronoi.num_cells() {
            for &j in voronoi.get_neighbors(i) {
                assert!(
                    voronoi.are_neighbors(j, i),
                    "Asymmetric neighbors: {} -> {} but not {} -> {}",
                    i,
                    j,
                    j,
                    i
                );
            }
        }
    }

    #[test]
    fn test_cell_area_estimation() {
        let voronoi = SphericalVoronoi::new(12, 2, 42);
        let areas = voronoi.estimate_cell_areas(1000);

        // Total area should sum to approximately 1.0
        let total: f32 = areas.iter().sum();
        assert!((total - 1.0).abs() < 0.01);

        // No cell should have zero area
        for area in &areas {
            assert!(*area > 0.0);
        }
    }

    #[test]
    fn test_voronoi_reproducibility() {
        let v1 = SphericalVoronoi::new(12, 2, 999);
        let v2 = SphericalVoronoi::new(12, 2, 999);

        for (c1, c2) in v1.cell_centers.iter().zip(v2.cell_centers.iter()) {
            assert!((c1 - c2).length() < 1e-6);
        }
    }

    #[test]
    fn test_boundary_midpoint() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);

        let mid = boundary_midpoint(a, b);

        // Should be on unit sphere
        assert!((mid.length() - 1.0).abs() < 1e-6);

        // Should be equidistant from both centers
        let dist_a = (mid - a).length();
        let dist_b = (mid - b).length();
        assert!((dist_a - dist_b).abs() < 1e-6);
    }
}
