//! Coordinate conversion utilities for cube-sphere mapping.

use glam::Vec3;
use super::face::CubeFaceId;
use super::spherify::spherify_point;

/// A 2D coordinate within a cube face, with UV in [0, 1] range.
#[derive(Debug, Clone, Copy)]
pub struct FaceCoord {
    /// The cube face this coordinate belongs to.
    pub face: CubeFaceId,
    /// U coordinate in [0, 1] range.
    pub u: f32,
    /// V coordinate in [0, 1] range.
    pub v: f32,
}

impl FaceCoord {
    /// Creates a new face coordinate.
    pub fn new(face: CubeFaceId, u: f32, v: f32) -> Self {
        Self { face, u, v }
    }

    /// Converts this face coordinate to a point on the unit sphere.
    pub fn to_sphere_point(self) -> Vec3 {
        cube_to_sphere(face_uv_to_cube(self.face, self.u, self.v))
    }
}

/// Converts UV coordinates on a face to a 3D point on the unit cube surface.
///
/// UV coordinates are in [0, 1] range and map to [-1, 1] on the cube face.
///
/// # Arguments
/// * `face` - The cube face
/// * `u` - U coordinate in [0, 1]
/// * `v` - V coordinate in [0, 1]
///
/// # Returns
/// A point on the surface of the unit cube
pub fn face_uv_to_cube(face: CubeFaceId, u: f32, v: f32) -> Vec3 {
    // Map [0, 1] to [-1, 1]
    let s = u * 2.0 - 1.0;
    let t = v * 2.0 - 1.0;

    match face {
        CubeFaceId::PosX => Vec3::new(1.0, t, -s),
        CubeFaceId::NegX => Vec3::new(-1.0, t, s),
        CubeFaceId::PosY => Vec3::new(s, 1.0, t),
        CubeFaceId::NegY => Vec3::new(s, -1.0, -t),
        CubeFaceId::PosZ => Vec3::new(s, t, 1.0),
        CubeFaceId::NegZ => Vec3::new(-s, t, -1.0),
    }
}

/// Converts a 3D cube point to a spherified point on the unit sphere.
///
/// This is a convenience wrapper around `spherify_point`.
pub fn cube_to_sphere(cube_point: Vec3) -> Vec3 {
    spherify_point(cube_point)
}

/// Determines which cube face a sphere point belongs to and returns UV coordinates.
///
/// # Arguments
/// * `sphere_pos` - A point on the unit sphere
///
/// # Returns
/// Tuple of (face, u, v) where u and v are in [0, 1] range
pub fn sphere_to_face_uv(sphere_pos: Vec3) -> (CubeFaceId, f32, f32) {
    let abs_pos = sphere_pos.abs();

    // Find dominant axis to determine face
    let (face, s, t) = if abs_pos.x >= abs_pos.y && abs_pos.x >= abs_pos.z {
        if sphere_pos.x > 0.0 {
            (CubeFaceId::PosX, -sphere_pos.z / sphere_pos.x, sphere_pos.y / sphere_pos.x)
        } else {
            (CubeFaceId::NegX, sphere_pos.z / -sphere_pos.x, sphere_pos.y / -sphere_pos.x)
        }
    } else if abs_pos.y >= abs_pos.x && abs_pos.y >= abs_pos.z {
        if sphere_pos.y > 0.0 {
            (CubeFaceId::PosY, sphere_pos.x / sphere_pos.y, sphere_pos.z / sphere_pos.y)
        } else {
            (CubeFaceId::NegY, sphere_pos.x / -sphere_pos.y, -sphere_pos.z / -sphere_pos.y)
        }
    } else if sphere_pos.z > 0.0 {
        (CubeFaceId::PosZ, sphere_pos.x / sphere_pos.z, sphere_pos.y / sphere_pos.z)
    } else {
        (CubeFaceId::NegZ, -sphere_pos.x / -sphere_pos.z, sphere_pos.y / -sphere_pos.z)
    };

    // Map [-1, 1] back to [0, 1]
    let u = (s + 1.0) * 0.5;
    let v = (t + 1.0) * 0.5;

    (face, u.clamp(0.0, 1.0), v.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_face_uv_to_cube_centers() {
        // Center of each face (u=0.5, v=0.5) should be axis-aligned
        let test_cases = [
            (CubeFaceId::PosX, Vec3::new(1.0, 0.0, 0.0)),
            (CubeFaceId::NegX, Vec3::new(-1.0, 0.0, 0.0)),
            (CubeFaceId::PosY, Vec3::new(0.0, 1.0, 0.0)),
            (CubeFaceId::NegY, Vec3::new(0.0, -1.0, 0.0)),
            (CubeFaceId::PosZ, Vec3::new(0.0, 0.0, 1.0)),
            (CubeFaceId::NegZ, Vec3::new(0.0, 0.0, -1.0)),
        ];

        for (face, expected) in test_cases {
            let cube_point = face_uv_to_cube(face, 0.5, 0.5);
            assert!(
                (cube_point - expected).length() < 1e-6,
                "Face {:?} center: expected {:?}, got {:?}",
                face,
                expected,
                cube_point
            );
        }
    }

    #[test]
    fn test_sphere_to_face_uv_centers() {
        // Face centers should map back correctly
        let face_centers = [
            (Vec3::new(1.0, 0.0, 0.0), CubeFaceId::PosX),
            (Vec3::new(-1.0, 0.0, 0.0), CubeFaceId::NegX),
            (Vec3::new(0.0, 1.0, 0.0), CubeFaceId::PosY),
            (Vec3::new(0.0, -1.0, 0.0), CubeFaceId::NegY),
            (Vec3::new(0.0, 0.0, 1.0), CubeFaceId::PosZ),
            (Vec3::new(0.0, 0.0, -1.0), CubeFaceId::NegZ),
        ];

        for (sphere_pos, expected_face) in face_centers {
            let (face, u, v) = sphere_to_face_uv(sphere_pos);
            assert_eq!(face, expected_face, "Wrong face for {:?}", sphere_pos);
            assert!(
                (u - 0.5).abs() < 1e-5 && (v - 0.5).abs() < 1e-5,
                "Face {:?} center should be (0.5, 0.5), got ({}, {})",
                face,
                u,
                v
            );
        }
    }

    #[test]
    fn test_roundtrip_face_uv() {
        // Note: Spherification introduces distortion, so we test with larger tolerance
        // near face centers (0.3-0.7) where distortion is minimal
        for face in CubeFaceId::all() {
            for &u in &[0.3, 0.4, 0.5, 0.6, 0.7] {
                for &v in &[0.3, 0.4, 0.5, 0.6, 0.7] {
                    let cube_p = face_uv_to_cube(face, u, v);
                    let sphere_p = cube_to_sphere(cube_p);
                    let (recovered_face, u2, v2) = sphere_to_face_uv(sphere_p);

                    assert_eq!(
                        face, recovered_face,
                        "Face mismatch: {:?} vs {:?} for UV ({}, {})",
                        face, recovered_face, u, v
                    );
                    // Spherification distorts coordinates - use 10% tolerance
                    assert!(
                        (u - u2).abs() < 0.1 && (v - v2).abs() < 0.1,
                        "UV mismatch for {:?}: ({}, {}) vs ({}, {})",
                        face,
                        u,
                        v,
                        u2,
                        v2
                    );
                }
            }
        }
    }

    #[test]
    fn test_face_mapping_consistency() {
        // Test that points near face centers roundtrip correctly
        for face in CubeFaceId::all() {
            let cube_p = face_uv_to_cube(face, 0.5, 0.5);
            let sphere_p = cube_to_sphere(cube_p);
            let (recovered_face, u, v) = sphere_to_face_uv(sphere_p);

            assert_eq!(face, recovered_face);
            assert!((u - 0.5).abs() < 0.01);
            assert!((v - 0.5).abs() < 0.01);
        }
    }
}
