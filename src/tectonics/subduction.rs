//! Subduction zone mechanics and uplift transfer functions.

use glam::Vec3;
use super::boundary::{PlateBoundary, BoundaryType};
use super::plate::{TectonicPlate, CrustType};

/// A subduction zone where one plate dives beneath another.
#[derive(Debug, Clone)]
pub struct SubductionZone {
    /// Index of the overriding plate (stays on surface).
    pub overriding_plate: usize,
    /// Index of the subducting plate (dives into mantle).
    pub subducting_plate: usize,
    /// Points along the trench line on unit sphere.
    pub trench_line: Vec<Vec3>,
    /// Subduction angle in radians (typically 30-60 degrees = 0.52-1.05 rad).
    pub subduction_angle: f32,
    /// Subduction rate (relative velocity magnitude).
    pub subduction_rate: f32,
}

impl SubductionZone {
    /// Returns the distance from the trench to the volcanic arc in km.
    pub fn arc_distance(&self) -> f32 {
        // Arc forms where subducting slab reaches ~100km depth
        // Distance = depth / sin(angle)
        100.0 / self.subduction_angle.sin()
    }

    /// Returns the direction from trench toward the overriding plate.
    ///
    /// This is the direction in which uplift effects propagate.
    pub fn uplift_direction(&self, plates: &[TectonicPlate]) -> Vec3 {
        let overriding = &plates[self.overriding_plate];
        let subducting = &plates[self.subducting_plate];
        (overriding.center - subducting.center).normalize()
    }
}

/// Detects subduction zones from convergent boundaries.
///
/// Subduction occurs when:
/// - Oceanic crust meets continental crust (oceanic always subducts)
/// - Two oceanic plates meet (older/denser one subducts)
///
/// # Arguments
/// * `boundary` - A convergent plate boundary
/// * `plates` - Array of tectonic plates
///
/// # Returns
/// `Some(SubductionZone)` if subduction is occurring, `None` otherwise
pub fn detect_subduction(
    boundary: &PlateBoundary,
    plates: &[TectonicPlate],
) -> Option<SubductionZone> {
    // Only convergent boundaries can have subduction
    if boundary.boundary_type != BoundaryType::Convergent {
        return None;
    }

    let plate_a = &plates[boundary.plate_a];
    let plate_b = &plates[boundary.plate_b];

    // Determine which plate subducts based on crust type and age
    let (overriding_idx, subducting_idx) = match (plate_a.crust_type, plate_b.crust_type) {
        // Oceanic subducts under continental
        (CrustType::Oceanic, CrustType::Continental) => {
            (boundary.plate_b, boundary.plate_a)
        }
        (CrustType::Continental, CrustType::Oceanic) => {
            (boundary.plate_a, boundary.plate_b)
        }
        // If both oceanic, older (denser) plate subducts
        (CrustType::Oceanic, CrustType::Oceanic) => {
            if plate_a.age > plate_b.age {
                (boundary.plate_b, boundary.plate_a)
            } else {
                (boundary.plate_a, boundary.plate_b)
            }
        }
        // Continental-continental collision is not subduction
        (CrustType::Continental, CrustType::Continental) => {
            return None;
        }
    };

    // Extract trench line from boundary segments
    let trench_line: Vec<Vec3> = boundary
        .segments
        .iter()
        .map(|s| s.start)
        .chain(boundary.segments.last().map(|s| s.end))
        .collect();

    // Calculate subduction angle based on plate properties
    // Steeper angles for faster subduction and younger plates
    let subducting_plate = &plates[subducting_idx];
    let base_angle = std::f32::consts::FRAC_PI_4; // 45 degrees base
    let age_factor = (subducting_plate.age / 100.0).min(2.0); // Older = steeper
    let velocity_factor = (boundary.relative_velocity / 5.0).min(1.5); // Faster = steeper

    let subduction_angle = (base_angle * (1.0 + age_factor * 0.2 + velocity_factor * 0.1))
        .clamp(0.35, 1.22); // 20-70 degrees

    Some(SubductionZone {
        overriding_plate: overriding_idx,
        subducting_plate: subducting_idx,
        trench_line,
        subduction_angle,
        subduction_rate: boundary.relative_velocity,
    })
}

/// Calculates uplift based on distance from the trench.
///
/// Implements a transfer function based on Cortial et al. (2019) that models:
/// - Trench depression at the boundary
/// - Forearc bulge
/// - Volcanic arc uplift
/// - Back-arc extension (optional depression)
///
/// # Arguments
/// * `distance_from_trench` - Distance in km (positive = toward overriding plate)
/// * `subduction_rate` - Rate of subduction (relative velocity)
/// * `subduction_angle` - Angle of subducting slab in radians
///
/// # Returns
/// Uplift value in km (positive = up, negative = down)
pub fn subduction_uplift(
    distance_from_trench: f32,
    subduction_rate: f32,
    subduction_angle: f32,
) -> f32 {
    let d = distance_from_trench;
    let v = subduction_rate;

    // Trench depth: depression right at the boundary
    // Maximum depth scales with subduction rate
    let trench_depth = -0.5 * v * 0.15 * (-d * d / 50.0).exp();

    // Forearc bulge: small uplift between trench and arc
    let forearc_dist = 50.0; // km from trench
    let forearc_width = 30.0; // km width of bulge
    let forearc_uplift = 0.2 * v * 0.1 * (-(d - forearc_dist).powi(2) / (forearc_width * forearc_width)).exp();

    // Volcanic arc: main uplift zone
    // Position depends on subduction angle (deeper angle = closer arc)
    let arc_distance = 100.0 / subduction_angle.sin();
    let arc_width = 50.0; // km width of arc
    let arc_uplift = 0.8 * v * 0.2 * (-(d - arc_distance).powi(2) / (arc_width * arc_width)).exp();

    // Back-arc: slight depression behind the arc
    let backarc_dist = arc_distance + 150.0;
    let backarc_width = 100.0;
    let backarc_depression = -0.1 * v * 0.05 * (-(d - backarc_dist).powi(2) / (backarc_width * backarc_width)).exp();

    // Combine all effects
    // Only apply effects on the overriding plate side (d > 0)
    if d < -20.0 {
        // On subducting plate, only show trench
        trench_depth
    } else {
        trench_depth + forearc_uplift + arc_uplift + backarc_depression
    }
}

/// Calculates the signed distance from a point to the nearest trench point.
///
/// # Arguments
/// * `pos` - Position on unit sphere
/// * `zone` - The subduction zone
/// * `plates` - Array of plates (to determine which side is overriding)
///
/// # Returns
/// Distance in radians, converted to approximate km assuming Earth radius.
/// Positive = toward overriding plate, negative = toward subducting plate.
pub fn distance_to_trench(
    pos: Vec3,
    zone: &SubductionZone,
    plates: &[TectonicPlate],
) -> f32 {
    // Find nearest trench point
    let mut min_dist = f32::MAX;
    let mut nearest_trench = zone.trench_line[0];

    for &trench_point in &zone.trench_line {
        let dist = great_circle_distance(pos, trench_point);
        if dist < min_dist {
            min_dist = dist;
            nearest_trench = trench_point;
        }
    }

    // Determine sign based on which side of trench the point is on
    let overriding_center = plates[zone.overriding_plate].center;
    let subducting_center = plates[zone.subducting_plate].center;

    let to_overriding = great_circle_distance(pos, overriding_center);
    let to_subducting = great_circle_distance(pos, subducting_center);

    // Convert from radians to approximate km (Earth radius ~6371 km)
    let dist_km = min_dist * 6371.0;

    if to_overriding < to_subducting {
        dist_km  // On overriding plate side
    } else {
        -dist_km // On subducting plate side
    }
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

    fn create_test_plate(id: usize, center: Vec3, crust_type: CrustType, age: f32) -> TectonicPlate {
        TectonicPlate {
            id,
            center,
            crust_type,
            angular_velocity: Vec3::new(0.0, 0.1, 0.0),
            area: 0.1,
            age,
        }
    }

    fn create_convergent_boundary(plate_a: usize, plate_b: usize) -> PlateBoundary {
        PlateBoundary {
            plate_a,
            plate_b,
            boundary_type: BoundaryType::Convergent,
            relative_velocity: 5.0,
            segments: vec![BoundarySegment {
                start: Vec3::new(0.0, 1.0, 0.0),
                end: Vec3::new(0.0, 0.9, 0.436),
                local_velocity: Vec3::new(0.1, 0.0, 0.0),
                boundary_type: BoundaryType::Convergent,
            }],
        }
    }

    #[test]
    fn test_oceanic_subducts_under_continental() {
        let plates = vec![
            create_test_plate(0, Vec3::new(-1.0, 0.0, 0.0), CrustType::Oceanic, 100.0),
            create_test_plate(1, Vec3::new(1.0, 0.0, 0.0), CrustType::Continental, 1000.0),
        ];

        let boundary = create_convergent_boundary(0, 1);
        let zone = detect_subduction(&boundary, &plates).unwrap();

        assert_eq!(zone.subducting_plate, 0); // Oceanic subducts
        assert_eq!(zone.overriding_plate, 1); // Continental overrides
    }

    #[test]
    fn test_older_oceanic_subducts() {
        let plates = vec![
            create_test_plate(0, Vec3::new(-1.0, 0.0, 0.0), CrustType::Oceanic, 150.0),
            create_test_plate(1, Vec3::new(1.0, 0.0, 0.0), CrustType::Oceanic, 50.0),
        ];

        let boundary = create_convergent_boundary(0, 1);
        let zone = detect_subduction(&boundary, &plates).unwrap();

        assert_eq!(zone.subducting_plate, 0); // Older oceanic subducts
        assert_eq!(zone.overriding_plate, 1); // Younger oceanic overrides
    }

    #[test]
    fn test_no_subduction_for_continental_collision() {
        let plates = vec![
            create_test_plate(0, Vec3::new(-1.0, 0.0, 0.0), CrustType::Continental, 1000.0),
            create_test_plate(1, Vec3::new(1.0, 0.0, 0.0), CrustType::Continental, 800.0),
        ];

        let boundary = create_convergent_boundary(0, 1);
        let zone = detect_subduction(&boundary, &plates);

        assert!(zone.is_none());
    }

    #[test]
    fn test_uplift_profile_trench() {
        // At the trench (d=0), there should be depression
        let uplift_at_trench = subduction_uplift(0.0, 5.0, 0.785);
        assert!(uplift_at_trench < 0.0, "Trench should have negative uplift");
    }

    #[test]
    fn test_uplift_profile_arc() {
        // At the volcanic arc, there should be uplift
        let arc_distance = 100.0 / 0.785_f32.sin(); // ~140 km
        let uplift_at_arc = subduction_uplift(arc_distance, 5.0, 0.785);
        assert!(uplift_at_arc > 0.0, "Arc should have positive uplift");
    }

    #[test]
    fn test_uplift_profile_decreases_far() {
        // Far from the trench, uplift should approach zero
        let uplift_far = subduction_uplift(500.0, 5.0, 0.785);
        assert!(uplift_far.abs() < 0.1, "Uplift should decay with distance");
    }

    #[test]
    fn test_arc_distance_calculation() {
        let zone = SubductionZone {
            overriding_plate: 0,
            subducting_plate: 1,
            trench_line: vec![Vec3::new(0.0, 1.0, 0.0)],
            subduction_angle: std::f32::consts::FRAC_PI_4, // 45 degrees
            subduction_rate: 5.0,
        };

        let arc_dist = zone.arc_distance();
        // At 45 degrees, arc should be at ~141 km (100 / sin(45°))
        assert!((arc_dist - 141.4).abs() < 1.0);
    }
}
