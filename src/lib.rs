//! Procedural Earth-like planet generator.
//!
//! This crate provides tools for generating realistic planetary terrain using
//! a cube-sphere geometry system with fractal noise-based heightmap generation.

pub mod geometry;
pub mod noise;
pub mod terrain;
pub mod export;
pub mod pipeline;
pub mod tectonics;
pub mod erosion;
pub mod climate;

pub use geometry::{CubeFaceId, FaceCoord};
pub use noise::FractalNoiseConfig;
pub use terrain::{CubeFace, Planet};
pub use pipeline::{GenerationStage, HeightmapStage, TectonicStage, ErosionStage, ClimateStage, Pipeline, StageConfig};
pub use tectonics::{TectonicConfig, TectonicPlate, CrustType, SphericalVoronoi};
pub use erosion::{ErosionConfig, OutletModel};
pub use climate::ClimateConfig;
