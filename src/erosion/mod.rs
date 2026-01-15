//! Erosion and hydrology pipeline (Phase 3).
//!
//! This module contains both GPU (wgpu compute) and CPU utilities to sculpt
//! terrain using hydraulic + thermal erosion, then extract drainage networks.

mod config;
pub mod cpu;
pub mod wgpu;

pub use config::{ErosionBackend, ErosionConfig, OutletModel};

