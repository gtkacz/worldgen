//! Headless wgpu context for compute workloads.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ErosionGpuError {
    #[error("No suitable GPU adapter found")]
    NoAdapter,
    #[error("Failed to request device: {0}")]
    RequestDevice(String),
}

/// Holds a wgpu device/queue used for erosion compute.
///
/// This is intentionally small; pipeline setup lives in `pipelines.rs`.
pub struct ErosionGpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl ErosionGpuContext {
    /// Create a headless wgpu device/queue suitable for compute.
    pub async fn new() -> Result<Self, ErosionGpuError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(ErosionGpuError::NoAdapter)?;

        let limits = wgpu::Limits::default();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("worldgen-erosion-device"),
                    // Needed for efficiently zero-initializing large textures at high resolutions.
                    required_features: wgpu::Features::CLEAR_TEXTURE,
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| ErosionGpuError::RequestDevice(e.to_string()))?;

        Ok(Self { device, queue })
    }
}

