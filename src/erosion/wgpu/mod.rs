//! wgpu compute implementation for erosion.

mod context;
mod pipelines;

pub use context::{ErosionGpuContext, ErosionGpuError};
pub use pipelines::{ErosionGpu, ErosionGpuOutputs};

