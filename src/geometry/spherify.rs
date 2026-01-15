//! Spherification algorithm for cube-to-sphere mapping.
//!
//! Uses the analytical spherification formula that provides better uniformity
//! than simple normalization, reducing area distortion at cube corners.

use glam::Vec3;

/// Transforms a point on the unit cube surface to the unit sphere.
///
/// This formula provides better uniformity than simple normalization,
/// significantly reducing area distortion at cube corners.
///
/// # Arguments
/// * `cube_pos` - A point on the surface of a unit cube (coordinates in [-1, 1])
///
/// # Returns
/// A point on the unit sphere (length = 1.0)
///
/// # Example
/// ```
/// use glam::Vec3;
/// use worldgen::geometry::spherify_point;
///
/// let cube_point = Vec3::new(1.0, 0.5, 0.5);
/// let sphere_point = spherify_point(cube_point);
/// assert!((sphere_point.length() - 1.0).abs() < 1e-6);
/// ```
pub fn spherify_point(cube_pos: Vec3) -> Vec3 {
    let x2 = cube_pos.x * cube_pos.x;
    let y2 = cube_pos.y * cube_pos.y;
    let z2 = cube_pos.z * cube_pos.z;

    // Analytical spherification formula
    // Provides better uniformity than simple normalization
    Vec3::new(
        cube_pos.x * (1.0 - y2 / 2.0 - z2 / 2.0 + y2 * z2 / 3.0).max(0.0).sqrt(),
        cube_pos.y * (1.0 - x2 / 2.0 - z2 / 2.0 + x2 * z2 / 3.0).max(0.0).sqrt(),
        cube_pos.z * (1.0 - x2 / 2.0 - y2 / 2.0 + x2 * y2 / 3.0).max(0.0).sqrt(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spherify_preserves_unit_length() {
        // Test points ON the cube surface (one coordinate must be ±1)
        let test_points = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.5, 0.5),
            Vec3::new(-1.0, 0.3, -0.7),
            Vec3::new(0.5, 1.0, -0.2),
            Vec3::new(0.8, 0.8, 1.0),
            Vec3::new(-1.0, -0.5, 0.5),
            Vec3::new(0.3, -1.0, 0.7),
        ];

        for p in test_points {
            let sphere_p = spherify_point(p);
            let len = sphere_p.length();
            assert!(
                (len - 1.0).abs() < 1e-5,
                "Point {:?} spherified to {:?} with length {} (expected 1.0)",
                p,
                sphere_p,
                len
            );
        }
    }

    #[test]
    fn test_spherify_cube_corners() {
        // Cube corners should map to sphere with length 1
        let corners = [
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(1.0, 1.0, -1.0),
        ];

        for corner in corners {
            let sphere_p = spherify_point(corner);
            let len = sphere_p.length();
            assert!(
                (len - 1.0).abs() < 1e-5,
                "Corner {:?} spherified to length {} (expected 1.0)",
                corner,
                len
            );
        }
    }

    #[test]
    fn test_spherify_face_centers() {
        // Face centers should remain unchanged (already on unit sphere)
        let face_centers = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
        ];

        for center in face_centers {
            let sphere_p = spherify_point(center);
            assert!(
                (sphere_p - center).length() < 1e-6,
                "Face center {:?} changed to {:?}",
                center,
                sphere_p
            );
        }
    }
}
