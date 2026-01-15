//! Oceanic ridge generation at divergent boundaries.

use glam::Vec3;
use super::boundary::{PlateBoundary, BoundaryType};
use super::plate::{TectonicPlate, CrustType};

/// An oceanic spreading ridge where new crust forms.
#[derive(Debug, Clone)]
pub struct OceanicRidge {
    /// Index of the first plate.
    pub plate_a: usize,
    /// Index of the second plate.
    pub plate_b: usize,
    /// Points along the ridge axis on unit sphere.
    pub ridge_axis: Vec<Vec3>,
    /// Half-spreading rate in cm/year equivalent.
    pub spreading_rate: f32,
    /// Whether this is a slow-spreading ridge (has axial valley).
    pub is_slow_spreading: bool,
}

impl OceanicRidge {
    /// Returns whether this ridge has an axial valley (slow spreading)
    /// or an axial high (fast spreading).
    pub fn has_axial_valley(&self) -> bool {
        self.is_slow_spreading
    }

    /// Returns the depth of the ridge crest in km (negative = below sea level).
    pub fn crest_depth(&self) -> f32 {
        // Fast ridges are slightly shallower due to more magma supply
        if self.is_slow_spreading {
            -2.7
        } else {
            -2.3
        }
    }

    /// Returns the direction perpendicular to the ridge.
    pub fn perpendicular_direction(&self, plates: &[TectonicPlate]) -> Vec3 {
        let center_a = plates[self.plate_a].center;
        let center_b = plates[self.plate_b].center;
        (center_b - center_a).normalize()
    }
}

/// Detects oceanic ridges from divergent boundaries.
///
/// Ridges form where two oceanic plates are moving apart.
///
/// # Arguments
/// * `boundary` - A divergent plate boundary
/// * `plates` - Array of tectonic plates
///
/// # Returns
/// `Some(OceanicRidge)` if a ridge is present, `None` otherwise
pub fn detect_ridge(
    boundary: &PlateBoundary,
    plates: &[TectonicPlate],
) -> Option<OceanicRidge> {
    // Only divergent boundaries can have ridges
    if boundary.boundary_type != BoundaryType::Divergent {
        return None;
    }

    let plate_a = &plates[boundary.plate_a];
    let plate_b = &plates[boundary.plate_b];

    // Both plates must be oceanic for a mid-ocean ridge
    // (Continental divergence creates rifts, not ridges)
    if plate_a.crust_type != CrustType::Oceanic || plate_b.crust_type != CrustType::Oceanic {
        return None;
    }

    // Extract ridge axis from boundary segments
    let ridge_axis: Vec<Vec3> = boundary
        .segments
        .iter()
        .map(|s| s.start)
        .chain(boundary.segments.last().map(|s| s.end))
        .collect();

    // Half-spreading rate (each plate moves at half the relative velocity)
    let spreading_rate = boundary.relative_velocity / 2.0;

    // Slow spreading: < 4 cm/year half-rate (e.g., Mid-Atlantic Ridge)
    // Fast spreading: > 4 cm/year half-rate (e.g., East Pacific Rise)
    let is_slow_spreading = spreading_rate < 4.0;

    Some(OceanicRidge {
        plate_a: boundary.plate_a,
        plate_b: boundary.plate_b,
        ridge_axis,
        spreading_rate,
        is_slow_spreading,
    })
}

/// Calculates seafloor depth based on distance from the ridge.
///
/// Uses the plate cooling model (Parsons & Sclater, 1977) which describes
/// how oceanic lithosphere cools and subsides as it moves away from the ridge.
///
/// # Arguments
/// * `distance_from_axis` - Distance from ridge axis in km
/// * `spreading_rate` - Half-spreading rate in cm/year
///
/// # Returns
/// Depth in km (negative values = below sea level)
pub fn ridge_depth_profile(distance_from_axis: f32, spreading_rate: f32) -> f32 {
    let d = distance_from_axis.abs();

    // Ridge crest depth (varies with spreading rate)
    let ridge_crest_depth = if spreading_rate < 4.0 {
        -2.7  // Slow spreading ridges are slightly deeper
    } else {
        -2.3  // Fast spreading ridges are shallower
    };

    // Age of crust at this distance (in Ma)
    // distance (km) / spreading_rate (cm/yr) * 0.01 (cm to km) * 1e-6 (yr to Ma)
    // Simplified: age = distance / (spreading_rate * 10)
    let age = if spreading_rate > 0.0 {
        d / (spreading_rate * 10.0)
    } else {
        0.0
    };

    // Thermal subsidence: depth increases with sqrt(age)
    // Based on the half-space cooling model
    // Subsidence rate: ~350 m/sqrt(Ma)
    let thermal_subsidence = 0.35 * age.sqrt();

    // Maximum depth for old oceanic crust (~6 km)
    // Beyond this, the lithosphere is thermally equilibrated
    let depth = ridge_crest_depth - thermal_subsidence;
    depth.max(-6.0)
}

/// Calculates the axial valley or high feature at the ridge crest.
///
/// Slow-spreading ridges have a characteristic axial valley (rift valley),
/// while fast-spreading ridges have an axial high.
///
/// # Arguments
/// * `distance_from_axis` - Distance from ridge axis in km
/// * `spreading_rate` - Half-spreading rate in cm/year
///
/// # Returns
/// Elevation modification in km (added to base depth profile)
pub fn axial_feature(distance_from_axis: f32, spreading_rate: f32) -> f32 {
    let d = distance_from_axis.abs();

    if spreading_rate < 4.0 {
        // Slow-spreading: axial valley
        // Valley width ~5-10 km, depth ~0.5-1.5 km
        let valley_width = 5.0;
        let valley_depth = 0.5 + (4.0 - spreading_rate) * 0.2;

        // Gaussian valley profile
        -valley_depth * (-d * d / (valley_width * valley_width)).exp()
    } else if spreading_rate > 6.0 {
        // Fast-spreading: axial high
        // High width ~2-3 km, elevation ~0.1-0.3 km
        let high_width = 2.0;
        let high_elevation = 0.1 + (spreading_rate - 6.0) * 0.05;

        // Gaussian high profile
        high_elevation * (-d * d / (high_width * high_width)).exp()
    } else {
        // Intermediate spreading: minimal axial feature
        0.0
    }
}

/// Calculates the total depth at a point near a ridge.
///
/// Combines the base depth profile with axial features.
///
/// # Arguments
/// * `distance_from_axis` - Distance from ridge axis in km
/// * `spreading_rate` - Half-spreading rate in cm/year
///
/// # Returns
/// Total depth in km (negative = below sea level)
pub fn total_ridge_depth(distance_from_axis: f32, spreading_rate: f32) -> f32 {
    let base_depth = ridge_depth_profile(distance_from_axis, spreading_rate);
    let axial = axial_feature(distance_from_axis, spreading_rate);
    base_depth + axial
}

/// Calculates the distance from a point to the nearest ridge axis point.
///
/// # Arguments
/// * `pos` - Position on unit sphere
/// * `ridge` - The oceanic ridge
///
/// # Returns
/// Distance in km
pub fn distance_to_ridge(pos: Vec3, ridge: &OceanicRidge) -> f32 {
    let mut min_dist = f32::MAX;

    for &axis_point in &ridge.ridge_axis {
        let dist = great_circle_distance(pos, axis_point);
        if dist < min_dist {
            min_dist = dist;
        }
    }

    // Convert from radians to km (Earth radius ~6371 km)
    min_dist * 6371.0
}

/// Computes the great circle distance between two points on a unit sphere.
fn great_circle_distance(a: Vec3, b: Vec3) -> f32 {
    let dot = a.dot(b).clamp(-1.0, 1.0);
    dot.acos()
}

/// Generates abyssal hill roughness pattern.
///
/// Abyssal hills are the most common landform on Earth, formed by
/// faulting and volcanic processes at the ridge. Their amplitude
/// decreases with spreading rate.
///
/// # Arguments
/// * `distance_from_axis` - Distance from ridge in km
/// * `spreading_rate` - Half-spreading rate in cm/year
/// * `along_axis_position` - Position along the ridge (for variation)
///
/// # Returns
/// Height variation in km
pub fn abyssal_hill_amplitude(distance_from_axis: f32, spreading_rate: f32) -> f32 {
    // Hills are more prominent on slow-spreading ridges
    let base_amplitude = if spreading_rate < 4.0 {
        0.3  // Slow spreading: ~300m hills
    } else {
        0.1  // Fast spreading: ~100m hills
    };

    // Amplitude decreases with distance as sediment covers hills
    let distance_factor = (-distance_from_axis.abs() / 500.0).exp();

    base_amplitude * distance_factor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tectonics::boundary::BoundarySegment;

    fn create_oceanic_plate(id: usize, center: Vec3) -> TectonicPlate {
        TectonicPlate {
            id,
            center,
            crust_type: CrustType::Oceanic,
            angular_velocity: Vec3::new(0.0, 0.1, 0.0),
            area: 0.1,
            age: 100.0,
        }
    }

    fn create_divergent_boundary(plate_a: usize, plate_b: usize, velocity: f32) -> PlateBoundary {
        PlateBoundary {
            plate_a,
            plate_b,
            boundary_type: BoundaryType::Divergent,
            relative_velocity: velocity,
            segments: vec![BoundarySegment {
                start: Vec3::new(0.0, 1.0, 0.0),
                end: Vec3::new(0.0, 0.9, 0.436),
                local_velocity: Vec3::new(-0.1, 0.0, 0.0),
                boundary_type: BoundaryType::Divergent,
            }],
        }
    }

    #[test]
    fn test_ridge_detection() {
        let plates = vec![
            create_oceanic_plate(0, Vec3::new(-1.0, 0.0, 0.0)),
            create_oceanic_plate(1, Vec3::new(1.0, 0.0, 0.0)),
        ];

        let boundary = create_divergent_boundary(0, 1, 6.0);
        let ridge = detect_ridge(&boundary, &plates);

        assert!(ridge.is_some());
        let ridge = ridge.unwrap();
        assert_eq!(ridge.spreading_rate, 3.0); // Half of 6.0
    }

    #[test]
    fn test_no_ridge_for_continental() {
        let plates = vec![
            TectonicPlate {
                id: 0,
                center: Vec3::new(-1.0, 0.0, 0.0),
                crust_type: CrustType::Continental,
                angular_velocity: Vec3::ZERO,
                area: 0.1,
                age: 500.0,
            },
            create_oceanic_plate(1, Vec3::new(1.0, 0.0, 0.0)),
        ];

        let boundary = create_divergent_boundary(0, 1, 6.0);
        let ridge = detect_ridge(&boundary, &plates);

        assert!(ridge.is_none());
    }

    #[test]
    fn test_slow_spreading_detection() {
        let plates = vec![
            create_oceanic_plate(0, Vec3::new(-1.0, 0.0, 0.0)),
            create_oceanic_plate(1, Vec3::new(1.0, 0.0, 0.0)),
        ];

        // Slow spreading (< 4 cm/yr half-rate)
        let slow_boundary = create_divergent_boundary(0, 1, 4.0); // 2 cm/yr half-rate
        let slow_ridge = detect_ridge(&slow_boundary, &plates).unwrap();
        assert!(slow_ridge.is_slow_spreading);

        // Fast spreading (> 4 cm/yr half-rate)
        let fast_boundary = create_divergent_boundary(0, 1, 12.0); // 6 cm/yr half-rate
        let fast_ridge = detect_ridge(&fast_boundary, &plates).unwrap();
        assert!(!fast_ridge.is_slow_spreading);
    }

    #[test]
    fn test_ridge_crest_elevation() {
        // At the ridge axis (d=0), depth should be at ridge crest level
        let depth_at_axis = ridge_depth_profile(0.0, 3.0);
        assert!(depth_at_axis > -3.0 && depth_at_axis < -2.0);
    }

    #[test]
    fn test_thermal_subsidence() {
        // Depth should increase with distance from ridge
        let depth_near = ridge_depth_profile(100.0, 3.0);
        let depth_far = ridge_depth_profile(1000.0, 3.0);
        assert!(depth_far < depth_near); // More negative = deeper
    }

    #[test]
    fn test_maximum_depth() {
        // Very old crust should not exceed ~6 km depth
        let depth_very_far = ridge_depth_profile(5000.0, 3.0);
        assert!(depth_very_far >= -6.1);
    }

    #[test]
    fn test_slow_ridge_axial_valley() {
        // Slow spreading ridge should have negative axial feature (valley)
        let axial = axial_feature(0.0, 2.0);
        assert!(axial < 0.0, "Slow ridge should have axial valley");
    }

    #[test]
    fn test_fast_ridge_axial_high() {
        // Fast spreading ridge should have positive axial feature (high)
        let axial = axial_feature(0.0, 8.0);
        assert!(axial > 0.0, "Fast ridge should have axial high");
    }

    #[test]
    fn test_abyssal_hills_decrease_with_rate() {
        let slow_amp = abyssal_hill_amplitude(100.0, 2.0);
        let fast_amp = abyssal_hill_amplitude(100.0, 8.0);
        assert!(slow_amp > fast_amp);
    }
}
