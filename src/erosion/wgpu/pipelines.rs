//! Compute pipelines for hydraulic + thermal erosion.

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};

use crate::erosion::ErosionConfig;
use crate::terrain::Planet;
use super::context::{ErosionGpuContext, ErosionGpuError};

/// Outputs produced by the GPU hydraulic erosion step.
#[derive(Debug, Clone)]
pub struct ErosionGpuOutputs {
    pub heights: Vec<f32>,
    pub water: Vec<f32>,
    pub sediment: Vec<f32>,
    /// Net height delta caused by hydraulic+thermal erosion (excluding depression filling).
    pub deposition: Vec<f32>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Params {
    resolution: u32,
    _pad0: [u32; 3],
    // p0: rainfall, evaporation, Ks, Kd
    p0: [f32; 4],
    // p1: Kc, dt, _, _
    p1: [f32; 4],
}

struct FieldTex {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
}

fn align_to(value: u32, alignment: u32) -> u32 {
    debug_assert!(alignment.is_power_of_two());
    (value + alignment - 1) & !(alignment - 1)
}

fn texture_desc(label: &str, resolution: u32, format: wgpu::TextureFormat) -> wgpu::TextureDescriptor<'_> {
    wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    }
}

fn create_field(ctx: &ErosionGpuContext, label: &str, resolution: u32, format: wgpu::TextureFormat) -> FieldTex {
    let tex = ctx.device.create_texture(&texture_desc(label, resolution, format));
    let view = tex.create_view(&wgpu::TextureViewDescriptor {
        label: Some(&format!("{}-view", label)),
        format: Some(format),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(6),
        usage: None,
    });
    FieldTex { tex, view }
}

fn pack_f32_layers_padded(resolution: u32, values: &[f32]) -> (Vec<u8>, u32) {
    let per_face = (resolution * resolution) as usize;
    assert_eq!(values.len(), per_face * 6);

    // wgpu requires bytes_per_row to be a multiple of 256 for buffer-texture copies.
    let bytes_per_row = resolution * 4;
    let padded_bpr = align_to(bytes_per_row, 256);
    let padded_row_floats = (padded_bpr / 4) as usize;

    let mut out = vec![0f32; padded_row_floats * (resolution as usize) * 6];
    for face in 0..6 {
        for y in 0..resolution as usize {
            let src_row_start = face * per_face + y * resolution as usize;
            let dst_row_start = face * (padded_row_floats * resolution as usize) + y * padded_row_floats;
            out[dst_row_start..dst_row_start + resolution as usize]
                .copy_from_slice(&values[src_row_start..src_row_start + resolution as usize]);
        }
    }

    (bytemuck::cast_slice(&out).to_vec(), padded_bpr)
}

fn pack_vec4_layers_padded(resolution: u32, values: &[[f32; 4]]) -> (Vec<u8>, u32) {
    let per_face = (resolution * resolution) as usize;
    assert_eq!(values.len(), per_face * 6);

    let bytes_per_row = resolution * 16;
    let padded_bpr = align_to(bytes_per_row, 256);
    let padded_row_vec4 = (padded_bpr / 16) as usize;

    let mut out = vec![[0f32; 4]; padded_row_vec4 * (resolution as usize) * 6];
    for face in 0..6 {
        for y in 0..resolution as usize {
            let src_row_start = face * per_face + y * resolution as usize;
            let dst_row_start = face * (padded_row_vec4 * resolution as usize) + y * padded_row_vec4;
            out[dst_row_start..dst_row_start + resolution as usize]
                .copy_from_slice(&values[src_row_start..src_row_start + resolution as usize]);
        }
    }

    (bytemuck::cast_slice(&out).to_vec(), padded_bpr)
}

fn unpack_f32_layers_padded(resolution: u32, padded_bpr: u32, bytes: &[u8]) -> Vec<f32> {
    let bytes_per_row = padded_bpr as usize;
    let rows_per_image = resolution as usize;
    let layer_stride = bytes_per_row * rows_per_image;
    assert!(bytes.len() >= layer_stride * 6);

    let mut out = vec![0f32; (resolution as usize) * (resolution as usize) * 6];
    for face in 0..6 {
        for y in 0..rows_per_image {
            let src = face * layer_stride + y * bytes_per_row;
            let row_f32: &[f32] = bytemuck::cast_slice(&bytes[src..src + bytes_per_row]);
            let dst_row_start = face * (resolution as usize) * (resolution as usize) + y * resolution as usize;
            out[dst_row_start..dst_row_start + resolution as usize]
                .copy_from_slice(&row_f32[..resolution as usize]);
        }
    }
    out
}

fn unpack_vec4_layers_padded(resolution: u32, padded_bpr: u32, bytes: &[u8]) -> Vec<[f32; 4]> {
    let bytes_per_row = padded_bpr as usize;
    let rows_per_image = resolution as usize;
    let layer_stride = bytes_per_row * rows_per_image;
    assert!(bytes.len() >= layer_stride * 6);

    let mut out = vec![[0f32; 4]; (resolution as usize) * (resolution as usize) * 6];
    for face in 0..6 {
        for y in 0..rows_per_image {
            let src = face * layer_stride + y * bytes_per_row;
            let row_v4: &[[f32; 4]] = bytemuck::cast_slice(&bytes[src..src + bytes_per_row]);
            let dst_row_start = face * (resolution as usize) * (resolution as usize) + y * resolution as usize;
            out[dst_row_start..dst_row_start + resolution as usize]
                .copy_from_slice(&row_v4[..resolution as usize]);
        }
    }
    out
}

pub struct ErosionGpu {
    ctx: ErosionGpuContext,
    bgl: wgpu::BindGroupLayout,
    rainfall: wgpu::ComputePipeline,
    flow: wgpu::ComputePipeline,
    water_update: wgpu::ComputePipeline,
    erosion_deposition: wgpu::ComputePipeline,
    transport: wgpu::ComputePipeline,
    evaporation: wgpu::ComputePipeline,
    thermal: wgpu::ComputePipeline,
}

impl ErosionGpu {
    pub fn new(ctx: ErosionGpuContext) -> Self {
        let shader_src = {
            let neighbors = include_str!("shaders/neighbors.wgsl");
            let hydraulic = include_str!("shaders/hydraulic.wgsl");
            format!("{neighbors}\n{hydraulic}")
        };
        let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("worldgen-erosion-hydraulic-wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_src)),
        });

        let bgl = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("worldgen-erosion-bgl"),
            entries: &[
                // Params
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(std::mem::size_of::<Params>() as u64).unwrap()),
                    },
                    count: None,
                },
                // Inputs
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                // Outputs
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("worldgen-erosion-pipeline-layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let rainfall = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rainfall"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("rainfall"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let flow = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("flow"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("flow"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let water_update = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("water_update"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("water_update"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let erosion_deposition = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("erosion_deposition"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("erosion_deposition"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let transport = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("transport"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("transport"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let evaporation = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("evaporation"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaporation"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let thermal = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("thermal"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("thermal"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            ctx,
            bgl,
            rainfall,
            flow,
            water_update,
            erosion_deposition,
            transport,
            evaporation,
            thermal,
        }
    }

    fn create_bind_group(
        &self,
        params: &wgpu::Buffer,
        height_in: &wgpu::TextureView,
        water_in: &wgpu::TextureView,
        sediment_in: &wgpu::TextureView,
        flux_in: &wgpu::TextureView,
        height_out: &wgpu::TextureView,
        water_out: &wgpu::TextureView,
        sediment_out: &wgpu::TextureView,
        flux_out: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("worldgen-erosion-bind-group"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(height_in),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(water_in),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(sediment_in),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(flux_in),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(height_out),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(water_out),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(sediment_out),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(flux_out),
                },
            ],
        })
    }

    fn dispatch(
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        resolution: u32,
    ) {
        let gx = (resolution + 7) / 8;
        let gy = (resolution + 7) / 8;
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("worldgen-erosion-pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.dispatch_workgroups(gx, gy, 6);
    }

    fn planet_to_layers(planet: &Planet) -> Vec<f32> {
        let res = planet.resolution();
        let per_face = (res * res) as usize;
        let mut out = vec![0f32; per_face * 6];
        for (i, face) in planet.faces.iter().enumerate() {
            out[i * per_face..(i + 1) * per_face].copy_from_slice(&face.heights);
        }
        out
    }

    fn layers_to_planet(planet: &mut Planet, heights: &[f32]) {
        let res = planet.resolution();
        let per_face = (res * res) as usize;
        for (i, face) in planet.faces.iter_mut().enumerate() {
            face.heights.copy_from_slice(&heights[i * per_face..(i + 1) * per_face]);
        }
    }

    fn upload_r32(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        tex: &wgpu::Texture,
        resolution: u32,
        values: &[f32],
        label: &str,
    ) {
        let (bytes, padded_bpr) = pack_f32_layers_padded(resolution, values);
        let buf = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}-upload-buffer")),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.ctx.queue.write_buffer(&buf, 0, &bytes);

        encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBuffer {
                buffer: &buf,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(resolution),
                },
            },
            wgpu::ImageCopyTexture {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 6,
            },
        );
    }

    fn upload_rgba32(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        tex: &wgpu::Texture,
        resolution: u32,
        values: &[[f32; 4]],
        label: &str,
    ) {
        let (bytes, padded_bpr) = pack_vec4_layers_padded(resolution, values);
        let buf = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}-upload-buffer")),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.ctx.queue.write_buffer(&buf, 0, &bytes);

        encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBuffer {
                buffer: &buf,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(resolution),
                },
            },
            wgpu::ImageCopyTexture {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 6,
            },
        );
    }

    fn readback_r32(&self, tex: &wgpu::Texture, resolution: u32, label: &str) -> Vec<f32> {
        let bytes_per_row = resolution * 4;
        let padded_bpr = align_to(bytes_per_row, 256);
        let layer_stride = (padded_bpr as u64) * (resolution as u64);
        let size = layer_stride * 6;

        let readback = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}-readback-buffer")),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("{label}-readback-encoder")),
        });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(resolution),
                },
            },
            wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 6,
            },
        );
        self.ctx.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // Block until the mapping is ready.
        self.ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let out = unpack_f32_layers_padded(resolution, padded_bpr, &data);
        drop(data);
        readback.unmap();
        out
    }

    /// Run GPU hydraulic erosion on the planet and write the heightfield back.
    pub fn run_hydraulic(&self, planet: &mut Planet, config: &ErosionConfig) -> Result<ErosionGpuOutputs, ErosionGpuError> {
        let resolution = planet.resolution();
        let total = (resolution as usize) * (resolution as usize) * 6;

        // Textures (ping-pong): 0=input, 1=output
        let mut height = [
            create_field(&self.ctx, "height-a", resolution, wgpu::TextureFormat::R32Float),
            create_field(&self.ctx, "height-b", resolution, wgpu::TextureFormat::R32Float),
        ];
        let mut water = [
            create_field(&self.ctx, "water-a", resolution, wgpu::TextureFormat::R32Float),
            create_field(&self.ctx, "water-b", resolution, wgpu::TextureFormat::R32Float),
        ];
        let mut sediment = [
            create_field(&self.ctx, "sediment-a", resolution, wgpu::TextureFormat::R32Float),
            create_field(&self.ctx, "sediment-b", resolution, wgpu::TextureFormat::R32Float),
        ];
        let mut flux = [
            create_field(&self.ctx, "flux-a", resolution, wgpu::TextureFormat::Rgba32Float),
            create_field(&self.ctx, "flux-b", resolution, wgpu::TextureFormat::Rgba32Float),
        ];

        let params = Params {
            resolution,
            _pad0: [0; 3],
            p0: [
                config.rainfall,
                config.evaporation,
                config.erosion_rate,
                config.deposition_rate,
            ],
            // talus ~ tan(angle)*cell_size. Use a crude cell_size tied to face resolution.
            p1: [
                config.sediment_capacity,
                1.0,
                config.angle_of_repose_rad.tan() * (1.0 / resolution as f32),
                config.thermal_strength,
            ],
        };
        let params_buf = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("worldgen-erosion-params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.ctx.queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        // Upload initial data
        let heights0 = Self::planet_to_layers(planet);
        let zeros = vec![0.0f32; total];
        let zeros_flux = vec![[0.0f32; 4]; total];

        let mut init = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("worldgen-erosion-init-encoder"),
        });
        self.upload_r32(&mut init, &height[0].tex, resolution, &heights0, "height");
        self.upload_r32(&mut init, &height[1].tex, resolution, &heights0, "height-copy");
        self.upload_r32(&mut init, &water[0].tex, resolution, &zeros, "water");
        self.upload_r32(&mut init, &water[1].tex, resolution, &zeros, "water-copy");
        self.upload_r32(&mut init, &sediment[0].tex, resolution, &zeros, "sediment");
        self.upload_r32(&mut init, &sediment[1].tex, resolution, &zeros, "sediment-copy");
        self.upload_rgba32(&mut init, &flux[0].tex, resolution, &zeros_flux, "flux");
        self.upload_rgba32(&mut init, &flux[1].tex, resolution, &zeros_flux, "flux-copy");
        self.ctx.queue.submit(Some(init.finish()));

        // Bind groups for ping-pong directions.
        let bind_0_to_1 = self.create_bind_group(
            &params_buf,
            &height[0].view,
            &water[0].view,
            &sediment[0].view,
            &flux[0].view,
            &height[1].view,
            &water[1].view,
            &sediment[1].view,
            &flux[1].view,
        );
        let bind_1_to_0 = self.create_bind_group(
            &params_buf,
            &height[1].view,
            &water[1].view,
            &sediment[1].view,
            &flux[1].view,
            &height[0].view,
            &water[0].view,
            &sediment[0].view,
            &flux[0].view,
        );

        let mut cur = 0usize;
        for _step in 0..config.hydraulic_steps {
            let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("worldgen-erosion-step-encoder"),
            });

            macro_rules! run_pass {
                ($pipeline:expr) => {{
                    let bg = if cur == 0 { &bind_0_to_1 } else { &bind_1_to_0 };
                    Self::dispatch(&mut encoder, $pipeline, bg, resolution);
                    cur ^= 1;
                }};
            }

            run_pass!(&self.rainfall);
            run_pass!(&self.flow);
            run_pass!(&self.water_update);
            run_pass!(&self.erosion_deposition);
            run_pass!(&self.transport);
            run_pass!(&self.evaporation);

            self.ctx.queue.submit(Some(encoder.finish()));
        }

        // Thermal relaxation
        if config.thermal_iterations > 0 {
            for _ in 0..config.thermal_iterations {
                let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("worldgen-thermal-encoder"),
                });
                let bg = if cur == 0 { &bind_0_to_1 } else { &bind_1_to_0 };
                Self::dispatch(&mut encoder, &self.thermal, bg, resolution);
                cur ^= 1;
                self.ctx.queue.submit(Some(encoder.finish()));
            }
        }

        // Read back from the current input textures (cur is the latest written-to side?).
        // After each pass we flip `cur`, so the most recent results live in `cur`.
        let h = self.readback_r32(&height[cur].tex, resolution, "height");
        let w = self.readback_r32(&water[cur].tex, resolution, "water");
        let s = self.readback_r32(&sediment[cur].tex, resolution, "sediment");

        Self::layers_to_planet(planet, &h);
        let deposition = if config.track_deposition {
            h.iter().zip(heights0.iter()).map(|(&h1, &h0)| h1 - h0).collect()
        } else {
            vec![0.0; h.len()]
        };

        Ok(ErosionGpuOutputs {
            heights: h,
            water: w,
            sediment: s,
            deposition,
        })
    }
}


