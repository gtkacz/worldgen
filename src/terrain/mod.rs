//! Terrain generation module.
//!
//! Provides the core Planet and CubeFace data structures for representing
//! planetary terrain data.

mod planet;
mod heightmap;

pub use planet::{Planet, CubeFace};
pub use heightmap::generate_heightmap;
