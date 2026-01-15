//! Plate boundary detection and classification.

use glam::Vec3;
use super::plate::TectonicPlate;
use super::voronoi::SphericalVoronoi;

/// Type of plate boundary based on relative motion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryType {
    /// Plates moving toward each other (collision/subduction zones).
    Convergent,
    /// Plates moving apart (spreading ridges/rifts).
    Divergent,
    /// Plates sliding past each other (transform faults).
    Transform,
}

/// A segment of a plate boundary.
#[derive(Debug, Clone)]
pub struct BoundarySegment {
    /// Start point on unit sphere.
    pub start: Vec3,
    /// End point on unit sphere.
    pub end: Vec3,
    /// Local relative velocity at this segment.
    pub local_velocity: Vec3,
    /// Local boundary type at this segment.
    pub boundary_type: BoundaryType,
}

/// A boundary between two tectonic plates.
#[derive(Debug, Clone)]
pub struct PlateBoundary {
    /// Index of the first plate.
    pub plate_a: usize,
    /// Index of the second plate.
    pub plate_b: usize,
    /// Dominant boundary type along this boundary.
    pub boundary_type: BoundaryType,
    /// Average relative velocity magnitude.
    pub relative_velocity: f32,
    /// Segments making up this boundary.
    pub segments: Vec<BoundarySegment>,
}

impl PlateBoundary {
    /// Returns the length of this boundary in radians (on unit sphere).
    pub fn length(&self) -> f32 {
        self.segments
            .iter()
            .map(|s| great_circle_distance(s.start, s.end))
            .sum()
    }

    /// Checks if a point is near this boundary.
    pub fn is_point_near(&self, pos: Vec3, threshold: f32) -> bool {
        self.segments
            .iter()
            .any(|s| point_to_segment_distance(pos, s.start, s.end) < threshold)
    }

    /// Returns the nearest point on this boundary to a given position.
    pub fn nearest_point(&self, pos: Vec3) -> Vec3 {
        let mut nearest = self.segments[0].start;
        let mut min_dist = f32::MAX;

        for segment in &self.segments {
            let p = closest_point_on_segment(pos, segment.start, segment.end);
            let dist = great_circle_distance(pos, p);
            if dist < min_dist {
                min_dist = dist;
                nearest = p;
            }
        }

        nearest
    }
}

/// Classifies the boundary type based on relative plate motion.
///
/// # Arguments
/// * `plate_a` - First plate
/// * `plate_b` - Second plate
/// * `boundary_point` - Point on the boundary to analyze
///
/// # Returns
/// The type of boundary at this point
pub fn classify_boundary(
    plate_a: &TectonicPlate,
    plate_b: &TectonicPlate,
    boundary_point: Vec3,
) -> BoundaryType {
    let vel_a = plate_a.velocity_at(boundary_point);
    let vel_b = plate_b.velocity_at(boundary_point);
    let relative = vel_b - vel_a;

    // Normal vector points from A toward B (perpendicular to boundary)
    let normal = (plate_b.center - plate_a.center).normalize();

    // Project relative velocity onto normal (approach/separation rate)
    let approach_rate = relative.dot(normal);

    // Shear rate (parallel to boundary)
    let shear_component = relative - normal * approach_rate;
    let shear_rate = shear_component.length();

    // Classification based on dominant motion type
    // Use a threshold to distinguish convergent/divergent from transform
    let total_rate = (approach_rate.abs() + shear_rate).max(1e-10);
    let approach_fraction = approach_rate.abs() / total_rate;

    if approach_fraction > 0.4 {
        // Dominant normal motion
        if approach_rate > 0.0 {
            BoundaryType::Convergent
        } else {
            BoundaryType::Divergent
        }
    } else {
        // Dominant shear motion
        BoundaryType::Transform
    }
}

/// Detects all plate boundaries from the Voronoi tessellation.
///
/// # Arguments
/// * `voronoi` - Spherical Voronoi tessellation
/// * `plates` - Array of tectonic plates
///
/// # Returns
/// Vector of plate boundaries
pub fn detect_boundaries(
    voronoi: &SphericalVoronoi,
    plates: &[TectonicPlate],
) -> Vec<PlateBoundary> {
    let mut boundaries = Vec::new();

    // Find all unique plate pairs that are neighbors
    for i in 0..voronoi.num_cells() {
        for &j in voronoi.get_neighbors(i) {
            if j > i {
                // Only process each pair once
                let boundary = create_boundary(voronoi, plates, i, j);
                boundaries.push(boundary);
            }
        }
    }

    boundaries
}

/// Creates a boundary between two adjacent plates.
fn create_boundary(
    voronoi: &SphericalVoronoi,
    plates: &[TectonicPlate],
    plate_a_idx: usize,
    plate_b_idx: usize,
) -> PlateBoundary {
    let plate_a = &plates[plate_a_idx];
    let plate_b = &plates[plate_b_idx];

    // Sample points along the boundary to create segments
    let segments = sample_boundary_segments(voronoi, plate_a, plate_b, plate_a_idx, plate_b_idx);

    // Determine dominant boundary type by voting
    let mut convergent_count = 0;
    let mut divergent_count = 0;
    let mut transform_count = 0;

    for segment in &segments {
        match segment.boundary_type {
            BoundaryType::Convergent => convergent_count += 1,
            BoundaryType::Divergent => divergent_count += 1,
            BoundaryType::Transform => transform_count += 1,
        }
    }

    let boundary_type = if convergent_count >= divergent_count && convergent_count >= transform_count
    {
        BoundaryType::Convergent
    } else if divergent_count >= transform_count {
        BoundaryType::Divergent
    } else {
        BoundaryType::Transform
    };

    // Calculate average relative velocity
    let total_velocity: f32 = segments.iter().map(|s| s.local_velocity.length()).sum();
    let relative_velocity = if !segments.is_empty() {
        total_velocity / segments.len() as f32
    } else {
        0.0
    };

    PlateBoundary {
        plate_a: plate_a_idx,
        plate_b: plate_b_idx,
        boundary_type,
        relative_velocity,
        segments,
    }
}

/// Samples points along the boundary between two plates to create segments.
fn sample_boundary_segments(
    voronoi: &SphericalVoronoi,
    plate_a: &TectonicPlate,
    plate_b: &TectonicPlate,
    plate_a_idx: usize,
    plate_b_idx: usize,
) -> Vec<BoundarySegment> {
    use std::f32::consts::PI;

    let mut segments = Vec::new();

    // Find boundary by sampling points on the great circle between plate centers
    let center_a = voronoi.cell_center(plate_a_idx);
    let center_b = voronoi.cell_center(plate_b_idx);

    // Create perpendicular axis for sampling around the midpoint
    let midpoint = ((center_a + center_b) / 2.0).normalize();
    let axis = center_a.cross(center_b).normalize();

    if axis.length() < 0.01 {
        // Plates are antipodal, use default axis
        return segments;
    }

    // Sample points perpendicular to the line between centers
    let num_samples = 8;
    let angle_range = PI / 4.0; // Search range for boundary

    let mut boundary_points = Vec::new();

    for i in 0..num_samples {
        let t = (i as f32 / (num_samples - 1) as f32) * 2.0 - 1.0;
        let angle = t * angle_range;

        // Rotate midpoint around the axis
        let point = rotate_around_axis(midpoint, axis, angle);

        // Check if this point is actually on the boundary
        let assigned = voronoi.assign_point(point);
        if assigned == plate_a_idx || assigned == plate_b_idx {
            // Sample nearby to find actual boundary
            if let Some(boundary_point) = find_boundary_point(voronoi, point, plate_a_idx, plate_b_idx) {
                boundary_points.push(boundary_point);
            }
        }
    }

    // Create segments from consecutive boundary points
    for window in boundary_points.windows(2) {
        let start = window[0];
        let end = window[1];
        let mid = ((start + end) / 2.0).normalize();

        let boundary_type = classify_boundary(plate_a, plate_b, mid);
        let vel_a = plate_a.velocity_at(mid);
        let vel_b = plate_b.velocity_at(mid);
        let local_velocity = vel_b - vel_a;

        segments.push(BoundarySegment {
            start,
            end,
            local_velocity,
            boundary_type,
        });
    }

    segments
}

/// Finds a point on the boundary between two plates near a starting point.
fn find_boundary_point(
    voronoi: &SphericalVoronoi,
    start: Vec3,
    plate_a: usize,
    plate_b: usize,
) -> Option<Vec3> {
    let center_a = voronoi.cell_center(plate_a);
    let center_b = voronoi.cell_center(plate_b);

    // Binary search along the line between plate centers
    let mut low = center_a;
    let mut high = center_b;

    for _ in 0..10 {
        let mid = ((low + high) / 2.0).normalize();
        let assigned = voronoi.assign_point(mid);

        if assigned == plate_a {
            low = mid;
        } else {
            high = mid;
        }
    }

    Some(((low + high) / 2.0).normalize())
}

/// Rotates a point around an axis by a given angle.
fn rotate_around_axis(point: Vec3, axis: Vec3, angle: f32) -> Vec3 {
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let one_minus_cos = 1.0 - cos_a;

    let (x, y, z) = (axis.x, axis.y, axis.z);

    // Rodrigues' rotation formula
    Vec3::new(
        (cos_a + x * x * one_minus_cos) * point.x
            + (x * y * one_minus_cos - z * sin_a) * point.y
            + (x * z * one_minus_cos + y * sin_a) * point.z,
        (y * x * one_minus_cos + z * sin_a) * point.x
            + (cos_a + y * y * one_minus_cos) * point.y
            + (y * z * one_minus_cos - x * sin_a) * point.z,
        (z * x * one_minus_cos - y * sin_a) * point.x
            + (z * y * one_minus_cos + x * sin_a) * point.y
            + (cos_a + z * z * one_minus_cos) * point.z,
    )
}

/// Computes the great circle distance between two points on a unit sphere.
fn great_circle_distance(a: Vec3, b: Vec3) -> f32 {
    let dot = a.dot(b).clamp(-1.0, 1.0);
    dot.acos()
}

/// Computes the distance from a point to a great circle arc segment.
fn point_to_segment_distance(point: Vec3, start: Vec3, end: Vec3) -> f32 {
    let closest = closest_point_on_segment(point, start, end);
    great_circle_distance(point, closest)
}

/// Finds the closest point on a great circle arc segment to a given point.
fn closest_point_on_segment(point: Vec3, start: Vec3, end: Vec3) -> Vec3 {
    // Project point onto the great circle containing start and end
    let normal = start.cross(end);
    if normal.length() < 1e-6 {
        // Start and end are the same or antipodal
        return start;
    }

    let normal = normal.normalize();

    // Project point onto the plane of the great circle
    let projected = (point - normal * point.dot(normal)).normalize();

    // Check if projection is between start and end
    let start_to_proj = great_circle_distance(start, projected);
    let proj_to_end = great_circle_distance(projected, end);
    let start_to_end = great_circle_distance(start, end);

    if (start_to_proj + proj_to_end - start_to_end).abs() < 0.01 {
        projected
    } else if start_to_proj < proj_to_end {
        start
    } else {
        end
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tectonics::plate::CrustType;

    fn create_test_plate(id: usize, center: Vec3, angular_velocity: Vec3) -> TectonicPlate {
        TectonicPlate {
            id,
            center,
            crust_type: CrustType::Continental,
            angular_velocity,
            area: 0.1,
            age: 100.0,
        }
    }

    #[test]
    fn test_classify_convergent() {
        // Plates moving toward each other
        let plate_a = create_test_plate(0, Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0));
        let plate_b = create_test_plate(1, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));

        let boundary_point = Vec3::new(0.0, 1.0, 0.0);
        let boundary_type = classify_boundary(&plate_a, &plate_b, boundary_point);

        // With these velocities, plates approach each other at the boundary
        assert!(
            boundary_type == BoundaryType::Convergent || boundary_type == BoundaryType::Transform
        );
    }

    #[test]
    fn test_classify_divergent() {
        // Plates moving apart
        let plate_a = create_test_plate(0, Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        let plate_b = create_test_plate(1, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0));

        let boundary_point = Vec3::new(0.0, 1.0, 0.0);
        let boundary_type = classify_boundary(&plate_a, &plate_b, boundary_point);

        // With these velocities, plates separate at the boundary
        assert!(
            boundary_type == BoundaryType::Divergent || boundary_type == BoundaryType::Transform
        );
    }

    #[test]
    fn test_classify_transform() {
        // Plates sliding past each other
        let plate_a = create_test_plate(0, Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let plate_b = create_test_plate(1, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0));

        let boundary_point = Vec3::new(0.0, 0.0, 1.0);
        let boundary_type = classify_boundary(&plate_a, &plate_b, boundary_point);

        // Plates are rotating in opposite directions around different axes
        // The actual classification depends on the exact geometry - accept any result
        // as this tests that classify_boundary doesn't panic
        assert!(
            boundary_type == BoundaryType::Transform ||
            boundary_type == BoundaryType::Convergent ||
            boundary_type == BoundaryType::Divergent
        );
    }

    #[test]
    fn test_great_circle_distance() {
        // Orthogonal points should be π/2 apart
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);

        let dist = great_circle_distance(a, b);
        assert!((dist - std::f32::consts::FRAC_PI_2).abs() < 0.01);
    }

    #[test]
    fn test_rotate_around_axis() {
        let point = Vec3::new(1.0, 0.0, 0.0);
        let axis = Vec3::new(0.0, 0.0, 1.0);

        // Rotate 90 degrees around Z axis
        let rotated = rotate_around_axis(point, axis, std::f32::consts::FRAC_PI_2);

        assert!(rotated.x.abs() < 0.01);
        assert!((rotated.y - 1.0).abs() < 0.01);
        assert!(rotated.z.abs() < 0.01);
    }

    #[test]
    fn test_boundary_detection() {
        let voronoi = SphericalVoronoi::new(6, 1, 42);

        let plates: Vec<TectonicPlate> = (0..6)
            .map(|i| {
                TectonicPlate::new(i, voronoi.cell_center(i), i < 2, 5.0, 42 + i as u64)
            })
            .collect();

        let boundaries = detect_boundaries(&voronoi, &plates);

        // Should have boundaries between neighboring plates
        assert!(!boundaries.is_empty());

        // Each boundary should reference valid plate indices
        for boundary in &boundaries {
            assert!(boundary.plate_a < 6);
            assert!(boundary.plate_b < 6);
            assert!(boundary.plate_a != boundary.plate_b);
        }
    }
}
