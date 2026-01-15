//! Noise generation module for terrain synthesis.
//!
//! Uses simdnoise for high-performance SIMD-accelerated noise generation.

mod fractal;

pub use fractal::{FractalNoiseConfig, sample_fractal_noise, sample_fractal_noise_batch};
