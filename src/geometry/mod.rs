//! Cube-sphere geometry module.
//!
//! Provides coordinate systems and transformations for mapping between
//! cube faces and spherical coordinates with minimal distortion.

mod face;
mod spherify;
mod cube_sphere;
pub mod neighbors;

pub use face::CubeFaceId;
pub use spherify::spherify_point;
pub use cube_sphere::{FaceCoord, face_uv_to_cube, sphere_to_face_uv, cube_to_sphere};
