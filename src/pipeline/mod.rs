//! Pipeline module for orchestrating planet generation stages.
//!
//! Provides a trait-based architecture for modular generation stages
//! that can be composed into a complete planet generation pipeline.

mod stage;

pub use stage::{
    GenerationStage, StageId, StageConfig, Pipeline, PipelineError,
    HeightmapStage, TectonicStage, ErosionStage, ClimateStage,
    BiomeStage,
};
