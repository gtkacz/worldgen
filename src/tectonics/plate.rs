//! Tectonic plate data structures.

use glam::Vec3;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Type of crustal material.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrustType {
    /// Continental crust: thick (~35km), light (2.7 g/cm³), old.
    Continental,
    /// Oceanic crust: thin (~7km), dense (3.0 g/cm³), younger.
    Oceanic,
}

impl CrustType {
    /// Returns the typical thickness of this crust type in km.
    pub fn thickness(&self) -> f32 {
        match self {
            CrustType::Continental => 35.0,
            CrustType::Oceanic => 7.0,
        }
    }

    /// Returns the typical density of this crust type in g/cm³.
    pub fn density(&self) -> f32 {
        match self {
            CrustType::Continental => 2.7,
            CrustType::Oceanic => 3.0,
        }
    }

    /// Returns the base elevation relative to sea level (km).
    /// Continental crust floats higher due to isostasy.
    pub fn base_elevation(&self) -> f32 {
        match self {
            CrustType::Continental => 0.5,  // Above sea level on average
            CrustType::Oceanic => -3.5,     // Below sea level
        }
    }
}

/// A tectonic plate on the planet surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TectonicPlate {
    /// Unique identifier for this plate.
    pub id: usize,
    /// Center point of the plate on the unit sphere.
    pub center: Vec3,
    /// Primary crust type of this plate.
    pub crust_type: CrustType,
    /// Angular velocity vector (rotation axis × angular speed).
    /// The magnitude represents angular speed in radians per unit time.
    pub angular_velocity: Vec3,
    /// Relative area of this plate (fraction of sphere surface).
    pub area: f32,
    /// Average age of the plate in Ma (million years).
    pub age: f32,
}

impl TectonicPlate {
    /// Creates a new tectonic plate with random properties.
    ///
    /// # Arguments
    /// * `id` - Unique plate identifier
    /// * `center` - Center point on unit sphere
    /// * `is_continental` - Whether this is a continental plate
    /// * `velocity_scale` - Scale factor for angular velocity
    /// * `seed` - Random seed for reproducible generation
    pub fn new(
        id: usize,
        center: Vec3,
        is_continental: bool,
        velocity_scale: f32,
        seed: u64,
    ) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(id as u64));

        // Generate random angular velocity axis
        let axis = Vec3::new(
            rng.random::<f32>() * 2.0 - 1.0,
            rng.random::<f32>() * 2.0 - 1.0,
            rng.random::<f32>() * 2.0 - 1.0,
        ).normalize();

        // Angular speed varies by plate
        let speed = velocity_scale * (0.5 + rng.random::<f32>() * 0.5);

        let crust_type = if is_continental {
            CrustType::Continental
        } else {
            CrustType::Oceanic
        };

        // Oceanic plates are generally younger than continental
        let age = if is_continental {
            500.0 + rng.random::<f32>() * 3000.0  // 500-3500 Ma
        } else {
            10.0 + rng.random::<f32>() * 190.0    // 10-200 Ma
        };

        Self {
            id,
            center,
            crust_type,
            angular_velocity: axis * speed,
            area: 0.0, // Set later by Voronoi calculation
            age,
        }
    }

    /// Calculates the linear velocity at a given point on the sphere.
    ///
    /// For a rigid rotation, the linear velocity is: v = ω × r
    /// where ω is the angular velocity and r is the position vector.
    ///
    /// # Arguments
    /// * `pos` - Position on the unit sphere
    ///
    /// # Returns
    /// Linear velocity vector tangent to the sphere at that point
    pub fn velocity_at(&self, pos: Vec3) -> Vec3 {
        self.angular_velocity.cross(pos)
    }

    /// Returns the angular speed (magnitude of angular velocity).
    pub fn angular_speed(&self) -> f32 {
        self.angular_velocity.length()
    }

    /// Returns the rotation axis (normalized angular velocity direction).
    pub fn rotation_axis(&self) -> Vec3 {
        let len = self.angular_velocity.length();
        if len > 1e-10 {
            self.angular_velocity / len
        } else {
            Vec3::Y // Default to Y-axis if stationary
        }
    }

    /// Checks if a point on the sphere is near the rotation axis
    /// (where velocity approaches zero).
    pub fn is_near_pole(&self, pos: Vec3, threshold: f32) -> bool {
        let axis = self.rotation_axis();
        let dot = pos.dot(axis).abs();
        dot > (1.0 - threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crust_type_properties() {
        assert!(CrustType::Continental.thickness() > CrustType::Oceanic.thickness());
        assert!(CrustType::Oceanic.density() > CrustType::Continental.density());
        assert!(CrustType::Continental.base_elevation() > CrustType::Oceanic.base_elevation());
    }

    #[test]
    fn test_plate_creation() {
        let center = Vec3::new(1.0, 0.0, 0.0);
        let plate = TectonicPlate::new(0, center, true, 5.0, 42);

        assert_eq!(plate.id, 0);
        assert_eq!(plate.crust_type, CrustType::Continental);
        assert!(plate.angular_speed() > 0.0);
        assert!(plate.age >= 500.0); // Continental plates are old
    }

    #[test]
    fn test_velocity_at_perpendicular() {
        let plate = TectonicPlate {
            id: 0,
            center: Vec3::new(1.0, 0.0, 0.0),
            crust_type: CrustType::Oceanic,
            angular_velocity: Vec3::new(0.0, 1.0, 0.0), // Rotate around Y
            area: 1.0,
            age: 50.0,
        };

        // At (1, 0, 0), rotation around Y produces velocity in -Z direction
        let vel = plate.velocity_at(Vec3::new(1.0, 0.0, 0.0));
        assert!(vel.x.abs() < 1e-6);
        assert!(vel.y.abs() < 1e-6);
        assert!(vel.z < 0.0);
    }

    #[test]
    fn test_velocity_at_rotation_axis() {
        let plate = TectonicPlate {
            id: 0,
            center: Vec3::new(0.0, 1.0, 0.0),
            crust_type: CrustType::Oceanic,
            angular_velocity: Vec3::new(0.0, 1.0, 0.0), // Rotate around Y
            area: 1.0,
            age: 50.0,
        };

        // At (0, 1, 0), which is on the rotation axis, velocity should be zero
        let vel = plate.velocity_at(Vec3::new(0.0, 1.0, 0.0));
        assert!(vel.length() < 1e-6);
    }

    #[test]
    fn test_is_near_pole() {
        let plate = TectonicPlate {
            id: 0,
            center: Vec3::new(0.0, 1.0, 0.0),
            crust_type: CrustType::Oceanic,
            angular_velocity: Vec3::new(0.0, 1.0, 0.0),
            area: 1.0,
            age: 50.0,
        };

        // Point on the rotation axis
        assert!(plate.is_near_pole(Vec3::new(0.0, 1.0, 0.0), 0.1));
        // Point far from rotation axis
        assert!(!plate.is_near_pole(Vec3::new(1.0, 0.0, 0.0), 0.1));
    }

    #[test]
    fn test_plate_reproducibility() {
        let center = Vec3::new(0.5, 0.5, 0.5).normalize();
        let plate1 = TectonicPlate::new(1, center, false, 5.0, 999);
        let plate2 = TectonicPlate::new(1, center, false, 5.0, 999);

        assert_eq!(plate1.angular_velocity, plate2.angular_velocity);
        assert_eq!(plate1.age, plate2.age);
    }
}
