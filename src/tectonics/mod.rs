//! Tectonic plate simulation system.
//!
//! This module implements a realistic tectonic plate simulation including:
//! - Spherical Voronoi tessellation for plate boundaries
//! - Plate movement with angular velocities
//! - Subduction zones with uplift transfer functions
//! - Continental collision orogeny (mountain building)
//! - Oceanic ridge generation at divergent boundaries

mod config;
mod plate;
mod voronoi;
pub mod boundary;
pub mod subduction;
pub mod orogeny;
pub mod ridge;

pub use config::TectonicConfig;
pub use plate::{TectonicPlate, CrustType};
pub use voronoi::SphericalVoronoi;
pub use boundary::{PlateBoundary, BoundaryType, BoundarySegment, detect_boundaries};
pub use subduction::{SubductionZone, subduction_uplift, detect_subduction, distance_to_trench};
pub use orogeny::{CollisionZone, collision_uplift, detect_collision, distance_to_suture, is_prowedge_side, himalayan_profile};
pub use ridge::{OceanicRidge, ridge_depth_profile, detect_ridge, total_ridge_depth, distance_to_ridge};
