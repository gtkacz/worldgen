//! Continental collision orogeny (mountain building).

use glam::Vec3;
use super::boundary::{PlateBoundary, BoundaryType};
use super::plate::{TectonicPlate, CrustType};

/// A zone of continental collision where mountains form.
#[derive(Debug, Clone)]
pub struct CollisionZone {
    /// Index of the first colliding plate.
    pub plate_a: usize,
    /// Index of the second colliding plate.
    pub plate_b: usize,
    /// Points along the collision suture line on unit sphere.
    pub collision_line: Vec<Vec3>,
    /// Rate of convergence (relative velocity magnitude).
    pub convergence_rate: f32,
    /// Accumulated strain/shortening over time (simplified as rate × age factor).
    pub accumulated_strain: f32,
}

impl CollisionZone {
    /// Returns the direction perpendicular to the collision zone.
    ///
    /// Mountains form symmetrically (or asymmetrically) along this direction.
    pub fn perpendicular_direction(&self, plates: &[TectonicPlate]) -> Vec3 {
        let center_a = plates[self.plate_a].center;
        let center_b = plates[self.plate_b].center;
        (center_b - center_a).normalize()
    }

    /// Returns the estimated maximum mountain height in km.
    pub fn max_mountain_height(&self) -> f32 {
        // Simplified model: height scales with accumulated strain
        // Real Himalayas: ~50 Ma of collision at ~5 cm/yr = ~2500 km shortening
        // Produces ~8 km mountains
        (self.accumulated_strain * 0.003).min(10.0)
    }

    /// Returns the estimated width of the mountain belt in km.
    pub fn mountain_width(&self) -> f32 {
        // Width scales with shortening
        200.0 + self.accumulated_strain * 0.5
    }
}

/// Detects collision zones from convergent boundaries between continental plates.
///
/// # Arguments
/// * `boundary` - A convergent plate boundary
/// * `plates` - Array of tectonic plates
///
/// # Returns
/// `Some(CollisionZone)` if continental collision is occurring, `None` otherwise
pub fn detect_collision(
    boundary: &PlateBoundary,
    plates: &[TectonicPlate],
) -> Option<CollisionZone> {
    // Only convergent boundaries can have collision
    if boundary.boundary_type != BoundaryType::Convergent {
        return None;
    }

    let plate_a = &plates[boundary.plate_a];
    let plate_b = &plates[boundary.plate_b];

    // Both plates must be continental for mountain-building collision
    if plate_a.crust_type != CrustType::Continental
        || plate_b.crust_type != CrustType::Continental
    {
        return None;
    }

    // Extract collision line from boundary segments
    let collision_line: Vec<Vec3> = boundary
        .segments
        .iter()
        .map(|s| s.start)
        .chain(boundary.segments.last().map(|s| s.end))
        .collect();

    // Estimate accumulated strain based on plate ages and convergence rate
    // Younger collision = less strain, older = more
    let avg_age = (plate_a.age + plate_b.age) / 2.0;
    let collision_duration = (avg_age * 0.1).min(100.0); // Assume collision for fraction of plate age
    let accumulated_strain = boundary.relative_velocity * collision_duration;

    Some(CollisionZone {
        plate_a: boundary.plate_a,
        plate_b: boundary.plate_b,
        collision_line,
        convergence_rate: boundary.relative_velocity,
        accumulated_strain,
    })
}

/// Calculates mountain height based on distance from the collision suture.
///
/// Uses a Gaussian profile to model the cross-section of a mountain belt.
///
/// # Arguments
/// * `distance_from_suture` - Distance in km (can be positive or negative)
/// * `convergence_rate` - Rate of plate convergence
/// * `collision_duration` - Duration of collision in Ma
///
/// # Returns
/// Uplift value in km
pub fn collision_uplift(
    distance_from_suture: f32,
    convergence_rate: f32,
    collision_duration: f32,
) -> f32 {
    let d = distance_from_suture;

    // Crustal shortening creates thickening
    let shortening = convergence_rate * collision_duration;

    // Mountain width scales with shortening
    let width = 200.0 + shortening * 0.5;
    let half_width_sq = (width / 2.0).powi(2);

    // Peak height from isostatic balance
    // Simplified model: height proportional to shortening
    let max_height = (shortening * 0.003).min(10.0);

    // Gaussian profile across orogen
    max_height * (-d * d / half_width_sq).exp()
}

/// Calculates asymmetric mountain profile (Himalayan-type).
///
/// Models the characteristic asymmetry of continent-continent collisions,
/// where one side (the "pro-wedge") is steeper than the other ("retro-wedge").
///
/// # Arguments
/// * `distance` - Signed distance from suture in km
/// * `toward_prowedge` - True if distance is toward the steeper side
/// * `convergence_rate` - Rate of plate convergence
/// * `collision_duration` - Duration of collision in Ma
///
/// # Returns
/// Uplift value in km
pub fn himalayan_profile(
    distance: f32,
    toward_prowedge: bool,
    convergence_rate: f32,
    collision_duration: f32,
) -> f32 {
    let d = distance.abs();

    // Asymmetric decay: steeper on pro-wedge side
    let decay = if toward_prowedge {
        100.0  // Steeper slope (like the Himalayan front)
    } else {
        300.0  // Gentler slope (like the Tibetan Plateau)
    };

    // Peak height scales with convergence and duration
    let shortening = convergence_rate * collision_duration;
    let peak_height = (shortening * 0.003).min(10.0);

    // Exponential decay profile
    peak_height * (-d / decay).exp()
}

/// Calculates the signed distance from a point to the nearest collision suture.
///
/// # Arguments
/// * `pos` - Position on unit sphere
/// * `zone` - The collision zone
/// * `plates` - Array of plates (to determine which side)
///
/// # Returns
/// Distance in km. Sign indicates which side of the suture.
pub fn distance_to_suture(
    pos: Vec3,
    zone: &CollisionZone,
    plates: &[TectonicPlate],
) -> f32 {
    // Find nearest suture point
    let mut min_dist = f32::MAX;

    for &suture_point in &zone.collision_line {
        let dist = great_circle_distance(pos, suture_point);
        if dist < min_dist {
            min_dist = dist;
        }
    }

    // Determine sign based on which plate the point is closer to
    let center_a = plates[zone.plate_a].center;
    let center_b = plates[zone.plate_b].center;

    let to_a = great_circle_distance(pos, center_a);
    let to_b = great_circle_distance(pos, center_b);

    // Convert from radians to approximate km (Earth radius ~6371 km)
    let dist_km = min_dist * 6371.0;

    if to_a < to_b {
        dist_km  // On plate A side
    } else {
        -dist_km // On plate B side
    }
}

/// Determines if a point is on the pro-wedge side of the collision.
///
/// The pro-wedge is typically the side with the younger or smaller plate
/// that is being "pushed" more actively.
pub fn is_prowedge_side(
    pos: Vec3,
    zone: &CollisionZone,
    plates: &[TectonicPlate],
) -> bool {
    let plate_a = &plates[zone.plate_a];
    let plate_b = &plates[zone.plate_b];

    // Pro-wedge is typically the side of the smaller/younger plate
    let prowedge_plate = if plate_a.area < plate_b.area || plate_a.age < plate_b.age {
        zone.plate_a
    } else {
        zone.plate_b
    };

    let prowedge_center = plates[prowedge_plate].center;
    let other_center = if prowedge_plate == zone.plate_a {
        plate_b.center
    } else {
        plate_a.center
    };

    let to_prowedge = great_circle_distance(pos, prowedge_center);
    let to_other = great_circle_distance(pos, other_center);

    to_prowedge < to_other
}

/// Computes the great circle distance between two points on a unit sphere.
fn great_circle_distance(a: Vec3, b: Vec3) -> f32 {
    let dot = a.dot(b).clamp(-1.0, 1.0);
    dot.acos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tectonics::boundary::BoundarySegment;

    fn create_continental_plate(id: usize, center: Vec3, age: f32, area: f32) -> TectonicPlate {
        TectonicPlate {
            id,
            center,
            crust_type: CrustType::Continental,
            angular_velocity: Vec3::new(0.0, 0.1, 0.0),
            area,
            age,
        }
    }

    fn create_convergent_boundary(plate_a: usize, plate_b: usize, velocity: f32) -> PlateBoundary {
        PlateBoundary {
            plate_a,
            plate_b,
            boundary_type: BoundaryType::Convergent,
            relative_velocity: velocity,
            segments: vec![BoundarySegment {
                start: Vec3::new(0.0, 1.0, 0.0),
                end: Vec3::new(0.0, 0.9, 0.436),
                local_velocity: Vec3::new(0.1, 0.0, 0.0),
                boundary_type: BoundaryType::Convergent,
            }],
        }
    }

    #[test]
    fn test_collision_detection() {
        let plates = vec![
            create_continental_plate(0, Vec3::new(-1.0, 0.0, 0.0), 1000.0, 0.2),
            create_continental_plate(1, Vec3::new(1.0, 0.0, 0.0), 800.0, 0.15),
        ];

        let boundary = create_convergent_boundary(0, 1, 5.0);
        let zone = detect_collision(&boundary, &plates);

        assert!(zone.is_some());
        let zone = zone.unwrap();
        assert_eq!(zone.plate_a, 0);
        assert_eq!(zone.plate_b, 1);
        assert!(zone.convergence_rate > 0.0);
    }

    #[test]
    fn test_no_collision_for_oceanic() {
        let plates = vec![
            TectonicPlate {
                id: 0,
                center: Vec3::new(-1.0, 0.0, 0.0),
                crust_type: CrustType::Oceanic,
                angular_velocity: Vec3::ZERO,
                area: 0.1,
                age: 100.0,
            },
            create_continental_plate(1, Vec3::new(1.0, 0.0, 0.0), 800.0, 0.15),
        ];

        let boundary = create_convergent_boundary(0, 1, 5.0);
        let zone = detect_collision(&boundary, &plates);

        assert!(zone.is_none());
    }

    #[test]
    fn test_collision_uplift_at_suture() {
        // At the suture (d=0), uplift should be maximum
        let uplift = collision_uplift(0.0, 5.0, 50.0);
        assert!(uplift > 0.0);

        // Uplift at suture should be greater than far away
        let uplift_far = collision_uplift(500.0, 5.0, 50.0);
        assert!(uplift > uplift_far);
    }

    #[test]
    fn test_collision_uplift_symmetric() {
        // Basic collision_uplift should be symmetric
        let uplift_pos = collision_uplift(100.0, 5.0, 50.0);
        let uplift_neg = collision_uplift(-100.0, 5.0, 50.0);
        assert!((uplift_pos - uplift_neg).abs() < 0.01);
    }

    #[test]
    fn test_himalayan_asymmetric() {
        // Himalayan profile should be asymmetric
        let uplift_pro = himalayan_profile(100.0, true, 5.0, 50.0);
        let uplift_retro = himalayan_profile(100.0, false, 5.0, 50.0);

        // Pro-wedge decays faster (steeper), so same distance should have less uplift
        assert!(uplift_pro < uplift_retro);
    }

    #[test]
    fn test_mountain_height_scales_with_convergence() {
        let zone_fast = CollisionZone {
            plate_a: 0,
            plate_b: 1,
            collision_line: vec![Vec3::Y],
            convergence_rate: 10.0,
            accumulated_strain: 500.0,
        };

        let zone_slow = CollisionZone {
            plate_a: 0,
            plate_b: 1,
            collision_line: vec![Vec3::Y],
            convergence_rate: 2.0,
            accumulated_strain: 100.0,
        };

        assert!(zone_fast.max_mountain_height() > zone_slow.max_mountain_height());
    }

    #[test]
    fn test_mountain_width_scales_with_shortening() {
        let zone_high = CollisionZone {
            plate_a: 0,
            plate_b: 1,
            collision_line: vec![Vec3::Y],
            convergence_rate: 5.0,
            accumulated_strain: 1000.0,
        };

        let zone_low = CollisionZone {
            plate_a: 0,
            plate_b: 1,
            collision_line: vec![Vec3::Y],
            convergence_rate: 5.0,
            accumulated_strain: 100.0,
        };

        assert!(zone_high.mountain_width() > zone_low.mountain_width());
    }
}
