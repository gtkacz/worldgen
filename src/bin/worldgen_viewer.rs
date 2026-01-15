//! Small interactive viewer for cube-face exports.
//!
//! Loads the 6 cube-face PNGs from disk and displays an assembled equirectangular
//! view in a window, with pan/zoom controls.

use std::error::Error;
use std::io;
use std::path::{Path, PathBuf};

use clap::Parser;
use glam::Vec2;
use image::GenericImageView;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    dpi::{PhysicalSize, Size},
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{WindowAttributes, WindowId},
};

use worldgen::geometry::CubeFaceId;

#[derive(Parser, Debug)]
#[command(name = "worldgen_viewer")]
#[command(about = "Interactive viewer for worldgen cube-face PNG exports")]
struct Args {
    /// Input directory containing face PNGs.
    #[arg(short, long, default_value = "./output")]
    input: PathBuf,

    /// Base name for height faces (expects `{name}_{face}.png`).
    #[arg(short, long, default_value = "planet")]
    name: String,

    /// Base name for biome faces (expects `{biomes_name}_{face}.png`).
    /// Defaults to `{name}_biomes`.
    #[arg(long)]
    biomes_name: Option<String>,
}

fn align_to(value: u32, alignment: u32) -> u32 {
    debug_assert!(alignment.is_power_of_two());
    (value + alignment - 1) & !(alignment - 1)
}

fn face_path(dir: &Path, base: &str, face: CubeFaceId) -> PathBuf {
    dir.join(format!("{}_{}.png", base, face.short_name()))
}

type AnyResult<T> = Result<T, Box<dyn Error>>;

fn other_err(msg: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::Other, msg.into()))
}

fn load_luma16_png(path: &Path) -> AnyResult<(u32, u32, Vec<u16>)> {
    let img = image::open(path)?;
    let (w, h) = img.dimensions();
    let l16 = img.to_luma16();
    Ok((w, h, l16.into_raw()))
}

fn load_rgb8_png(path: &Path) -> AnyResult<(u32, u32, Vec<u8>)> {
    let img = image::open(path)?;
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8();
    Ok((w, h, rgb.into_raw()))
}

fn pack_u16_rows_padded(width: u32, height: u32, pixels: &[u16]) -> (Vec<u8>, u32) {
    let bytes_per_row = width * 2;
    let padded_bpr = align_to(bytes_per_row, 256);
    let padded_row_u16 = (padded_bpr / 2) as usize;
    let w = width as usize;
    let h = height as usize;
    assert_eq!(pixels.len(), w * h);

    let mut out_u16 = vec![0u16; padded_row_u16 * h];
    for y in 0..h {
        let src = y * w;
        let dst = y * padded_row_u16;
        out_u16[dst..dst + w].copy_from_slice(&pixels[src..src + w]);
    }
    (bytemuck::cast_slice(&out_u16).to_vec(), padded_bpr)
}

fn pack_rgba8_rows_padded(width: u32, height: u32, rgba: &[u8]) -> (Vec<u8>, u32) {
    let bytes_per_row = width * 4;
    let padded_bpr = align_to(bytes_per_row, 256);
    let padded_row_bytes = padded_bpr as usize;
    let w_bytes = (width as usize) * 4;
    let h = height as usize;
    assert_eq!(rgba.len(), w_bytes * h);

    let mut out = vec![0u8; padded_row_bytes * h];
    for y in 0..h {
        let src = y * w_bytes;
        let dst = y * padded_row_bytes;
        out[dst..dst + w_bytes].copy_from_slice(&rgba[src..src + w_bytes]);
    }
    (out, padded_bpr)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    center_lon: f32,
    center_lat: f32,
    zoom: f32,
    mode: u32,       // 0=height, 1=biomes
    has_biomes: u32, // 0/1
    _pad0: [u32; 2],
    aspect: f32,
    _pad1: [f32; 3],
}

impl Uniforms {
    fn new(aspect: f32, has_biomes: bool) -> Self {
        Self {
            center_lon: 0.0,
            center_lat: 0.0,
            zoom: 1.0,
            mode: 0,
            has_biomes: if has_biomes { 1 } else { 0 },
            _pad0: [0; 2],
            aspect,
            _pad1: [0.0; 3],
        }
    }
}

fn lon_lat_from_screen(uv: Vec2, center_lon: f32, center_lat: f32, zoom: f32, aspect: f32) -> (f32, f32) {
    // Keep correct equirect proportions: full-map width/height should be 2:1
    let lon = center_lon + (uv.x - 0.5) * (std::f32::consts::PI * aspect) / zoom;
    let lat = center_lat + (0.5 - uv.y) * (std::f32::consts::PI) / zoom;
    let lat = lat.clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
    (lon, lat)
}

fn main() -> AnyResult<()> {
    let args = Args::parse();
    let biomes_base = args.biomes_name.clone().unwrap_or_else(|| format!("{}_biomes", args.name));

    // Load height faces (L16) from disk.
    let mut height_layers: Vec<Vec<u16>> = Vec::with_capacity(6);
    let mut face_w = 0u32;
    let mut face_h = 0u32;
    for face in CubeFaceId::all() {
        let p = face_path(&args.input, &args.name, face);
        let (w, h, px) = load_luma16_png(&p)?;
        if face_w == 0 {
            face_w = w;
            face_h = h;
        } else if w != face_w || h != face_h {
            return Err(other_err(format!(
                "Mismatched face dimensions: {:?} is {}x{}, expected {}x{}",
                p, w, h, face_w, face_h
            )));
        }
        height_layers.push(px);
    }

    // Try to load biome faces (Rgb8). If any are missing, disable biomes.
    let mut biomes_rgba_layers: Vec<Vec<u8>> = Vec::with_capacity(6);
    let mut has_biomes = true;
    for face in CubeFaceId::all() {
        let p = face_path(&args.input, &biomes_base, face);
        match load_rgb8_png(&p) {
            Ok((w, h, rgb)) => {
                if w != face_w || h != face_h {
                    return Err(other_err(format!(
                        "Mismatched biome face dimensions: {:?} is {}x{}, expected {}x{}",
                        p, w, h, face_w, face_h
                    )));
                }
                // Expand to RGBA for the GPU.
                let mut rgba = vec![0u8; (w as usize) * (h as usize) * 4];
                for i in 0..(w as usize * h as usize) {
                    rgba[i * 4 + 0] = rgb[i * 3 + 0];
                    rgba[i * 4 + 1] = rgb[i * 3 + 1];
                    rgba[i * 4 + 2] = rgb[i * 3 + 2];
                    rgba[i * 4 + 3] = 255;
                }
                biomes_rgba_layers.push(rgba);
            }
            Err(_) => {
                has_biomes = false;
                biomes_rgba_layers.clear();
                break;
            }
        }
    }

    let event_loop = EventLoop::new()?;

    let mut app = App {
        face_w,
        face_h,
        height_layers,
        biomes_layers: if has_biomes { Some(biomes_rgba_layers) } else { None },
        window: None,
        window_id: None,
        state: None,
        dragging: false,
        last_cursor: None,
    };

    event_loop.run_app(&mut app)?;
    Ok(())
}

struct App {
    face_w: u32,
    face_h: u32,
    height_layers: Vec<Vec<u16>>,
    biomes_layers: Option<Vec<Vec<u8>>>,

    window: Option<&'static winit::window::Window>,
    window_id: Option<WindowId>,
    state: Option<GpuState>,

    dragging: bool,
    last_cursor: Option<Vec2>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window = event_loop
            .create_window(
                WindowAttributes::default()
                    .with_title("worldgen viewer")
                    .with_inner_size(Size::Physical(PhysicalSize::new(960u32, 540u32))),
            )
            .expect("failed to create window");

        // Leak the window so we can hold a `'static` reference for wgpu surface lifetime.
        let window: &'static winit::window::Window = Box::leak(Box::new(window));
        self.window_id = Some(window.id());
        self.window = Some(window);

        let gpu = pollster::block_on(GpuState::new(
            window,
            self.face_w,
            self.face_h,
            &self.height_layers,
            self.biomes_layers.as_ref(),
        ))
        .expect("failed to initialize GPU state");

        self.state = Some(gpu);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if Some(window_id) != self.window_id {
            return;
        }
        let Some(state) = self.state.as_mut() else { return; };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size),
            WindowEvent::RedrawRequested => {
                if let Err(e) = state.render() {
                    match e {
                        wgpu::SurfaceError::Lost => state.reconfigure_surface(),
                        wgpu::SurfaceError::OutOfMemory => event_loop.exit(),
                        _ => {}
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Digit1) => state.uniforms.mode = 0,
                        PhysicalKey::Code(KeyCode::Digit2) => {
                            if state.uniforms.has_biomes != 0 {
                                state.uniforms.mode = 1
                            }
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseInput { state: s, button: MouseButton::Left, .. } => {
                self.dragging = s == ElementState::Pressed;
                if !self.dragging {
                    self.last_cursor = None;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let pos = Vec2::new(position.x as f32, position.y as f32);
                if self.dragging {
                    if let Some(prev) = self.last_cursor {
                        state.pan_pixels(pos - prev);
                    }
                }
                self.last_cursor = Some(pos);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll_y = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => (p.y as f32) / 120.0,
                };
                if scroll_y.abs() > 0.0001 {
                    state.zoom(scroll_y, self.last_cursor);
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(w) = self.window {
            w.request_redraw();
        }
    }
}

struct GpuState {
    window_size: PhysicalSize<u32>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    height_tex: wgpu::Texture,
    height_view: wgpu::TextureView,
    biomes_tex: Option<wgpu::Texture>,
    biomes_view: Option<wgpu::TextureView>,
    sampler: wgpu::Sampler,

    uniforms: Uniforms,
    uniforms_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
}

impl GpuState {
    async fn new(
        window: &'static winit::window::Window,
        face_w: u32,
        face_h: u32,
        height_layers: &[Vec<u16>],
        biomes_layers: Option<&Vec<Vec<u8>>>,
    ) -> AnyResult<Self> {
        let window_size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window)?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| other_err("No suitable GPU adapter found"))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("worldgen-viewer-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: window_size.width.max(1),
            height: window_size.height.max(1),
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let height_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("height-tex-array"),
            size: wgpu::Extent3d { width: face_w, height: face_h, depth_or_array_layers: 6 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let height_view = height_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("height-view"),
            format: Some(wgpu::TextureFormat::R16Unorm),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(6),
            usage: None,
        });

        // Upload height layers with padding (for alignment).
        for (layer, pixels) in height_layers.iter().enumerate() {
            let (bytes, padded_bpr) = pack_u16_rows_padded(face_w, face_h, pixels);
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &height_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: layer as u32 },
                    aspect: wgpu::TextureAspect::All,
                },
                &bytes,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(face_h),
                },
                wgpu::Extent3d { width: face_w, height: face_h, depth_or_array_layers: 1 },
            );
        }

        let (biomes_tex, biomes_view, has_biomes) = if let Some(layers) = biomes_layers {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("biomes-tex-array"),
                size: wgpu::Extent3d { width: face_w, height: face_h, depth_or_array_layers: 6 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor {
                label: Some("biomes-view"),
                format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(6),
                usage: None,
            });
            for (layer, rgba) in layers.iter().enumerate() {
                let (bytes, padded_bpr) = pack_rgba8_rows_padded(face_w, face_h, rgba);
                queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: layer as u32 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &bytes,
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_bpr),
                        rows_per_image: Some(face_h),
                    },
                    wgpu::Extent3d { width: face_w, height: face_h, depth_or_array_layers: 1 },
                );
            }
            (Some(tex), Some(view), true)
        } else {
            (None, None, false)
        };

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("viewer-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let aspect = (config.width as f32) / (config.height as f32);
        let uniforms = Uniforms::new(aspect, has_biomes);
        let uniforms_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("viewer-uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewer-shader"),
            source: wgpu::ShaderSource::Wgsl(VIEWER_WGSL.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewer-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // Biomes texture is optional; when missing, bind a dummy 1x1 texture.
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(std::mem::size_of::<Uniforms>() as u64).unwrap()),
                    },
                    count: None,
                },
            ],
        });

        let dummy_biomes = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dummy-biomes"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 6 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let dummy_view = dummy_biomes.create_view(&wgpu::TextureViewDescriptor {
            label: Some("dummy-biomes-view"),
            format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(6),
            usage: None,
        });

        let biomes_view_ref = biomes_view.as_ref().unwrap_or(&dummy_view);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewer-bind-group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&height_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(biomes_view_ref) },
                wgpu::BindGroupEntry { binding: 3, resource: uniforms_buf.as_entire_binding() },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewer-pipeline-layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewer-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Ok(Self {
            window_size,
            surface,
            device,
            queue,
            config,
            height_tex,
            height_view,
            biomes_tex,
            biomes_view,
            sampler,
            uniforms,
            uniforms_buf,
            bind_group,
            pipeline,
        })
    }

    fn reconfigure_surface(&mut self) {
        self.config.width = self.window_size.width.max(1);
        self.config.height = self.window_size.height.max(1);
        self.surface.configure(&self.device, &self.config);
        self.uniforms.aspect = (self.config.width as f32) / (self.config.height as f32);
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        self.window_size = size;
        self.reconfigure_surface();
    }

    fn pan_pixels(&mut self, delta: Vec2) {
        let w = self.config.width.max(1) as f32;
        let h = self.config.height.max(1) as f32;
        let aspect = self.uniforms.aspect;
        let lon_delta = (delta.x / w) * (std::f32::consts::PI * aspect) / self.uniforms.zoom;
        let lat_delta = (-delta.y / h) * (std::f32::consts::PI) / self.uniforms.zoom;
        self.uniforms.center_lon += lon_delta;
        self.uniforms.center_lat = (self.uniforms.center_lat + lat_delta)
            .clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
    }

    fn zoom(&mut self, scroll_y: f32, cursor: Option<Vec2>) {
        let old_zoom = self.uniforms.zoom;
        let factor = 1.1_f32.powf(scroll_y);
        let new_zoom = (old_zoom * factor).clamp(0.25, 128.0);

        // Zoom about cursor by adjusting center so the cursor stays on the same lon/lat.
        if let Some(c) = cursor {
            let w = self.config.width.max(1) as f32;
            let h = self.config.height.max(1) as f32;
            let uv = Vec2::new(c.x / w, c.y / h);
            let aspect = self.uniforms.aspect;
            let (lon0, lat0) = lon_lat_from_screen(uv, self.uniforms.center_lon, self.uniforms.center_lat, old_zoom, aspect);
            let (lon1, lat1) = lon_lat_from_screen(uv, self.uniforms.center_lon, self.uniforms.center_lat, new_zoom, aspect);
            self.uniforms.center_lon += lon0 - lon1;
            self.uniforms.center_lat = (self.uniforms.center_lat + (lat0 - lat1))
                .clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
        }

        self.uniforms.zoom = new_zoom;
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.queue.write_buffer(&self.uniforms_buf, 0, bytemuck::bytes_of(&self.uniforms));

        let frame = self.surface.get_current_texture()?;
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("viewer-encoder") });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewer-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }
}

const VIEWER_WGSL: &str = r#"
struct Uniforms {
  center_lon: f32,
  center_lat: f32,
  zoom: f32,
  mode: u32,
  has_biomes: u32,
  _pad0: vec2<u32>,
  aspect: f32,
  _pad1: vec3<f32>,
};

@group(0) @binding(0) var height_tex: texture_2d_array<f32>;
@group(0) @binding(1) var height_samp: sampler;
@group(0) @binding(2) var biomes_tex: texture_2d_array<f32>;
@group(0) @binding(3) var<uniform> u: Uniforms;

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
  // Fullscreen triangle.
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>( 3.0,  1.0),
    vec2<f32>(-1.0,  1.0)
  );
  var uvs = array<vec2<f32>, 3>(
    vec2<f32>(0.0, 2.0),
    vec2<f32>(2.0, 0.0),
    vec2<f32>(0.0, 0.0)
  );
  var out: VsOut;
  out.pos = vec4<f32>(positions[i], 0.0, 1.0);
  out.uv = uvs[i];
  return out;
}

fn wrap_lon(lon: f32) -> f32 {
  let pi = 3.141592653589793;
  let two_pi = 6.283185307179586;
  var x = lon;
  if (x > pi) {
    x = x - two_pi * floor((x + pi) / two_pi);
  }
  if (x < -pi) {
    x = x + two_pi * floor((-x + pi) / two_pi);
  }
  // After the math above, it should be in [-pi, pi] but clamp defensively.
  return clamp(x, -pi, pi);
}

fn lat_lon_to_dir(lat: f32, lon: f32) -> vec3<f32> {
  // lon=0 points to +Z
  let slon = sin(lon);
  let clon = cos(lon);
  let slat = sin(lat);
  let clat = cos(lat);
  return vec3<f32>(clat * slon, slat, clat * clon);
}

// Matches `sphere_to_face_uv` in Rust (`src/geometry/cube_sphere.rs`).
// Returns (layer, u, v).
fn sphere_to_face_uv(p: vec3<f32>) -> vec3<f32> {
  let ap = abs(p);
  var face: f32;
  var s: f32;
  var t: f32;

  if (ap.x >= ap.y && ap.x >= ap.z) {
    if (p.x > 0.0) {
      face = 0.0; // PosX
      s = -p.z / p.x;
      t =  p.y / p.x;
    } else {
      face = 1.0; // NegX
      s =  p.z / (-p.x);
      t =  p.y / (-p.x);
    }
  } else if (ap.y >= ap.x && ap.y >= ap.z) {
    if (p.y > 0.0) {
      face = 2.0; // PosY
      s =  p.x / p.y;
      t =  p.z / p.y;
    } else {
      face = 3.0; // NegY
      s =  p.x / (-p.y);
      t = -p.z / (-p.y);
    }
  } else if (p.z > 0.0) {
    face = 4.0; // PosZ
    s = p.x / p.z;
    t = p.y / p.z;
  } else {
    face = 5.0; // NegZ
    s = -p.x / (-p.z);
    t =  p.y / (-p.z);
  }

  let uu = clamp((s + 1.0) * 0.5, 0.0, 1.0);
  let vv = clamp((t + 1.0) * 0.5, 0.0, 1.0);
  return vec3<f32>(face, uu, vv);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  let pi = 3.141592653589793;

  // Map screen UV to lon/lat with correct proportions (full globe is 2:1).
  let lon = wrap_lon(u.center_lon + (in.uv.x - 0.5) * (pi * u.aspect) / u.zoom);
  let lat = clamp(u.center_lat + (0.5 - in.uv.y) * (pi) / u.zoom, -pi * 0.5, pi * 0.5);
  let dir = lat_lon_to_dir(lat, lon);
  let fuv = sphere_to_face_uv(normalize(dir));
  let layer = u32(fuv.x + 0.5);
  let uv = vec2<f32>(fuv.y, fuv.z);

  if (u.mode == 1u && u.has_biomes != 0u) {
    let c = textureSampleLevel(biomes_tex, height_samp, vec3<f32>(uv, f32(layer)), 0.0);
    return vec4<f32>(c.rgb, 1.0);
  }

  let h = textureSampleLevel(height_tex, height_samp, vec3<f32>(uv, f32(layer)), 0.0).r;
  return vec4<f32>(vec3<f32>(h), 1.0);
}
"#;

