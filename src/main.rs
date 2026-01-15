//! Worldgen CLI - Procedural planet generator.
//!
//! Generate Earth-like planetary terrain using cube-sphere geometry
//! and fractal noise-based heightmap generation.

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use std::time::Instant;

use worldgen::noise::FractalNoiseConfig;
use worldgen::terrain::Planet;
use worldgen::export::{
    export_planet_png, export_planet_raw,
    export_planet_plate_map, export_planet_boundary_map,
    PngExportOptions, RawFormat, PlateMapOptions,
    export_face_scalar_png_f32, export_face_mask_png_u8,
};
use worldgen::pipeline::{Pipeline, StageConfig, HeightmapStage, TectonicStage, ErosionStage, ClimateStage};
use worldgen::tectonics::TectonicConfig;
use worldgen::erosion::{ErosionConfig, OutletModel};
use worldgen::climate::{ClimateConfig, compute_coast_distance_km, precompute_lat_lon, compute_month};

/// Procedural Earth-like planet generator.
#[derive(Parser)]
#[command(name = "worldgen")]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a new planet heightmap.
    Generate {
        /// Per-face resolution in pixels (e.g., 512, 1024, 2048).
        #[arg(short, long, default_value = "512")]
        resolution: u32,

        /// Random seed for reproducible generation.
        #[arg(short, long)]
        seed: Option<u64>,

        /// Output directory for generated files.
        #[arg(short, long, default_value = "./output")]
        output: PathBuf,

        /// Base name for output files.
        #[arg(short, long, default_value = "planet")]
        name: String,

        /// Export format.
        #[arg(short, long, default_value = "png")]
        format: ExportFormat,

        /// Number of noise octaves (4-10).
        #[arg(long, default_value = "6")]
        octaves: u8,

        /// Base noise frequency.
        #[arg(long, default_value = "2.0")]
        frequency: f32,

        /// Frequency multiplier per octave (lacunarity).
        #[arg(long, default_value = "2.0")]
        lacunarity: f32,

        /// Amplitude decay per octave (persistence).
        #[arg(long, default_value = "0.5")]
        persistence: f32,

        /// Use Earth-like preset for terrain generation.
        #[arg(long)]
        earth_like: bool,

        // Tectonic options
        /// Number of tectonic plates (8-20 typical).
        #[arg(long, default_value = "12")]
        num_plates: usize,

        /// Fraction of surface that is continental (0.0-1.0).
        #[arg(long, default_value = "0.35")]
        continental_fraction: f32,

        /// Skip tectonic simulation (faster, less realistic).
        #[arg(long)]
        skip_tectonics: bool,

        /// Export plate visualization maps.
        #[arg(long)]
        plate_map: bool,

        /// Export boundary type maps (convergent/divergent/transform).
        #[arg(long)]
        boundary_map: bool,

        // Erosion options
        /// Skip erosion pipeline (hydraulic + thermal + rivers).
        #[arg(long)]
        skip_erosion: bool,

        /// Number of hydraulic erosion timesteps.
        #[arg(long, default_value = "200")]
        erosion_steps: u32,

        /// Rainfall per erosion step.
        #[arg(long, default_value = "0.01")]
        rainfall: f32,

        /// Evaporation factor per erosion step (0-1).
        #[arg(long, default_value = "0.02")]
        evaporation: f32,

        /// Thermal erosion iterations.
        #[arg(long, default_value = "150")]
        thermal_iters: u32,

        /// Angle of repose in degrees.
        #[arg(long, default_value = "35")]
        angle_of_repose_deg: f32,

        /// Thermal relaxation strength (0-1).
        #[arg(long, default_value = "0.25")]
        thermal_strength: f32,

        /// Sea level used as Priority-Flood outlet threshold.
        #[arg(long, default_value = "0.0")]
        sea_level: f32,

        /// River threshold in contributing cells (accumulation).
        #[arg(long, default_value = "500")]
        river_threshold: u32,

        /// Export water depth maps.
        #[arg(long)]
        water_map: bool,

        /// Export sediment maps.
        #[arg(long)]
        sediment_map: bool,

        /// Export flow accumulation maps.
        #[arg(long)]
        flow_map: bool,

        /// Export river mask maps.
        #[arg(long)]
        river_map: bool,

        /// Export net deposition/erosion maps (height delta from erosion backend).
        #[arg(long)]
        deposition_map: bool,

        // Climate options
        /// Skip climate simulation (Phase 4).
        #[arg(long)]
        skip_climate: bool,

        /// Axial tilt in degrees (affects seasonality).
        #[arg(long, default_value = "23.44")]
        axial_tilt_deg: f32,

        /// Moisture advection iterations per month (higher = wetter/smoother, slower).
        #[arg(long, default_value = "64")]
        climate_iters: u32,

        /// Export 12 monthly climate maps (temperature + precipitation), streamed to disk.
        #[arg(long)]
        export_climate_monthly: bool,
    },

    /// Display information about a planet configuration.
    Info {
        /// Per-face resolution in pixels.
        #[arg(short, long, default_value = "512")]
        resolution: u32,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum ExportFormat {
    /// 16-bit PNG (universal compatibility).
    Png,
    /// 16-bit RAW little-endian (Unity).
    Raw,
    /// 32-bit float RAW (high precision).
    RawFloat,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            resolution,
            seed,
            output,
            name,
            format,
            octaves,
            frequency,
            lacunarity,
            persistence,
            earth_like,
            num_plates,
            continental_fraction,
            skip_tectonics,
            plate_map,
            boundary_map,
            skip_erosion,
            erosion_steps,
            rainfall,
            evaporation,
            thermal_iters,
            angle_of_repose_deg,
            thermal_strength,
            sea_level,
            river_threshold,
            water_map,
            sediment_map,
            flow_map,
            river_map,
            deposition_map,
            skip_climate,
            axial_tilt_deg,
            climate_iters,
            export_climate_monthly,
        } => {
            run_generate(
                resolution,
                seed,
                output,
                name,
                format,
                octaves,
                frequency,
                lacunarity,
                persistence,
                earth_like,
                num_plates,
                continental_fraction,
                skip_tectonics,
                plate_map,
                boundary_map,
                skip_erosion,
                erosion_steps,
                rainfall,
                evaporation,
                thermal_iters,
                angle_of_repose_deg,
                thermal_strength,
                sea_level,
                river_threshold,
                water_map,
                sediment_map,
                flow_map,
                river_map,
                deposition_map,
                skip_climate,
                axial_tilt_deg,
                climate_iters,
                export_climate_monthly,
            );
        }
        Commands::Info { resolution } => {
            run_info(resolution);
        }
    }
}

fn run_generate(
    resolution: u32,
    seed: Option<u64>,
    output: PathBuf,
    name: String,
    format: ExportFormat,
    octaves: u8,
    frequency: f32,
    lacunarity: f32,
    persistence: f32,
    earth_like: bool,
    num_plates: usize,
    continental_fraction: f32,
    skip_tectonics: bool,
    plate_map: bool,
    boundary_map: bool,
    skip_erosion: bool,
    erosion_steps: u32,
    rainfall: f32,
    evaporation: f32,
    thermal_iters: u32,
    angle_of_repose_deg: f32,
    thermal_strength: f32,
    sea_level: f32,
    river_threshold: u32,
    water_map: bool,
    sediment_map: bool,
    flow_map: bool,
    river_map: bool,
    deposition_map: bool,
    skip_climate: bool,
    axial_tilt_deg: f32,
    climate_iters: u32,
    export_climate_monthly: bool,
) {
    // Validate parameters
    if resolution < 16 || resolution > 8192 {
        eprintln!("Error: Resolution must be between 16 and 8192");
        std::process::exit(1);
    }

    if octaves < 1 || octaves > 16 {
        eprintln!("Error: Octaves must be between 1 and 16");
        std::process::exit(1);
    }

    if num_plates < 4 || num_plates > 50 {
        eprintln!("Error: Number of plates must be between 4 and 50");
        std::process::exit(1);
    }

    if continental_fraction < 0.0 || continental_fraction > 1.0 {
        eprintln!("Error: Continental fraction must be between 0.0 and 1.0");
        std::process::exit(1);
    }

    // Generate seed if not provided
    let seed = seed.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    });

    println!("Worldgen - Procedural Planet Generator");
    println!("======================================");
    println!("Resolution: {}x{} per face", resolution, resolution);
    println!("Seed: {}", seed);
    println!("Output: {}", output.display());

    let start = Instant::now();

    // Create noise configuration
    let noise_config = if earth_like {
        println!("Preset: Earth-like");
        FractalNoiseConfig::earth_like(seed as i32)
    } else {
        FractalNoiseConfig {
            octaves,
            frequency,
            lacunarity,
            persistence,
            seed: seed as i32,
        }
    };

    // Create planet
    println!("\nCreating planet...");
    let mut planet = Planet::earth_like(resolution, seed);

    // Create and run pipeline
    println!("Running generation pipeline...");
    let stage_config = StageConfig::with_noise(noise_config);
    let mut pipeline = Pipeline::new(stage_config);
    pipeline.add_stage(HeightmapStage);

    // Add tectonic stage if not skipped
    if !skip_tectonics {
        let mut tectonic_config = TectonicConfig::earth_like(seed);
        tectonic_config.num_plates = num_plates;
        tectonic_config.continental_fraction = continental_fraction;
        pipeline.add_stage(TectonicStage::new(tectonic_config));
        println!("Tectonic simulation enabled: {} plates", num_plates);
    } else {
        println!("Tectonic simulation: SKIPPED");
    }

    // Add erosion stage if not skipped
    if !skip_erosion {
        let keep_intermediates = water_map || sediment_map || flow_map || river_map;
        let erosion_config = ErosionConfig {
            hydraulic_steps: erosion_steps,
            rainfall,
            evaporation,
            thermal_iterations: thermal_iters,
            angle_of_repose_rad: angle_of_repose_deg.to_radians(),
            thermal_strength,
            outlet_model: OutletModel::SeaLevel { sea_level },
            river_accum_threshold: river_threshold,
            keep_intermediates,
            track_deposition: deposition_map,
            ..Default::default()
        };
        pipeline.add_stage(ErosionStage::new(erosion_config));
        println!("Erosion pipeline enabled: hydraulic_steps={}", erosion_steps);
    } else {
        println!("Erosion pipeline: SKIPPED");
    }

    // Add climate stage (Phase 4) unless skipped.
    let mut climate_cfg = ClimateConfig::earth_like();
    climate_cfg.sea_level = sea_level;
    climate_cfg.axial_tilt_deg = axial_tilt_deg;
    climate_cfg.iterations = climate_iters;

    if !skip_climate {
        pipeline.add_stage(ClimateStage::new(climate_cfg.clone()));
        println!(
            "Climate simulation enabled: axial_tilt_deg={}, iters={}",
            axial_tilt_deg, climate_iters
        );
    } else {
        println!("Climate simulation: SKIPPED");
    }

    pipeline
        .run_with_callbacks(
            &mut planet,
            |name, i, total| {
                println!("  [{}/{}] Starting: {}", i + 1, total, name);
            },
            |name, i, total| {
                println!("  [{}/{}] Completed: {}", i + 1, total, name);
            },
        )
        .unwrap_or_else(|e| {
            eprintln!("Error during generation: {}", e);
            std::process::exit(1);
        });

    let gen_time = start.elapsed();
    println!("Generation completed in {:.2?}", gen_time);

    // Get height range for export
    let (min_h, max_h) = planet.height_range();
    println!("Height range: [{:.4}, {:.4}]", min_h, max_h);

    // Export
    println!("\nExporting heightmaps...");
    let export_start = Instant::now();

    std::fs::create_dir_all(&output).unwrap_or_else(|e| {
        eprintln!("Error creating output directory: {}", e);
        std::process::exit(1);
    });

    match format {
        ExportFormat::Png => {
            let options = PngExportOptions {
                min_height: min_h,
                max_height: max_h,
                ..Default::default()
            };
            export_planet_png(&planet, &output, &name, &options).unwrap_or_else(|e| {
                eprintln!("Error exporting PNG: {}", e);
                std::process::exit(1);
            });
            println!("  Exported 6 PNG files: {}_*.png", name);
        }
        ExportFormat::Raw => {
            export_planet_raw(&planet, &output, &name, RawFormat::R16LittleEndian, min_h, max_h)
                .unwrap_or_else(|e| {
                    eprintln!("Error exporting RAW: {}", e);
                    std::process::exit(1);
                });
            println!("  Exported 6 RAW files (R16): {}_*.raw", name);
        }
        ExportFormat::RawFloat => {
            export_planet_raw(&planet, &output, &name, RawFormat::R32Float, min_h, max_h)
                .unwrap_or_else(|e| {
                    eprintln!("Error exporting RAW: {}", e);
                    std::process::exit(1);
                });
            println!("  Exported 6 RAW files (R32 float): {}_*.raw", name);
        }
    }

    // Export plate maps if requested and tectonics was run
    if plate_map && !skip_tectonics {
        let plate_map_name = format!("{}_plates", name);
        let plate_options = PlateMapOptions::default();
        export_planet_plate_map(&planet, &output, &plate_map_name, &plate_options)
            .unwrap_or_else(|e| {
                eprintln!("Error exporting plate map: {}", e);
                std::process::exit(1);
            });
        println!("  Exported 6 plate map files: {}_*.png", plate_map_name);
    }

    if boundary_map && !skip_tectonics {
        let boundary_map_name = format!("{}_boundaries", name);
        export_planet_boundary_map(&planet, &output, &boundary_map_name)
            .unwrap_or_else(|e| {
                eprintln!("Error exporting boundary map: {}", e);
                std::process::exit(1);
            });
        println!("  Exported 6 boundary map files: {}_*.png", boundary_map_name);
    }

    // Export erosion-related maps if requested.
    if !skip_erosion && (water_map || sediment_map || flow_map || river_map || deposition_map) {
        let res = planet.resolution();

        if water_map {
            let mut min_v = f32::MAX;
            let mut max_v = f32::MIN;
            for face in &planet.faces {
                if let Some(w) = &face.water {
                    for &v in w {
                        min_v = min_v.min(v);
                        max_v = max_v.max(v);
                    }
                }
            }
            let min_v = if min_v.is_finite() { min_v } else { 0.0 };
            let max_v = if max_v.is_finite() && max_v > min_v { max_v } else { min_v + 1e-6 };

            for face in &planet.faces {
                if let Some(w) = &face.water {
                    let filename = format!("{}_water_{}.png", name, face.id.short_name());
                    let path = output.join(filename);
                    export_face_scalar_png_f32(
                        res,
                        w,
                        &path,
                        min_v,
                        max_v,
                        image::codecs::png::CompressionType::Default,
                        image::codecs::png::FilterType::Adaptive,
                    ).unwrap_or_else(|e| {
                        eprintln!("Error exporting water map: {}", e);
                        std::process::exit(1);
                    });
                }
            }
            println!("  Exported water maps: {}_water_*.png", name);
        }

        if sediment_map {
            let mut min_v = f32::MAX;
            let mut max_v = f32::MIN;
            for face in &planet.faces {
                if let Some(s) = &face.sediment {
                    for &v in s {
                        min_v = min_v.min(v);
                        max_v = max_v.max(v);
                    }
                }
            }
            let min_v = if min_v.is_finite() { min_v } else { 0.0 };
            let max_v = if max_v.is_finite() && max_v > min_v { max_v } else { min_v + 1e-6 };

            for face in &planet.faces {
                if let Some(s) = &face.sediment {
                    let filename = format!("{}_sediment_{}.png", name, face.id.short_name());
                    let path = output.join(filename);
                    export_face_scalar_png_f32(
                        res,
                        s,
                        &path,
                        min_v,
                        max_v,
                        image::codecs::png::CompressionType::Default,
                        image::codecs::png::FilterType::Adaptive,
                    ).unwrap_or_else(|e| {
                        eprintln!("Error exporting sediment map: {}", e);
                        std::process::exit(1);
                    });
                }
            }
            println!("  Exported sediment maps: {}_sediment_*.png", name);
        }

        if flow_map {
            let mut max_a: u32 = 0;
            for face in &planet.faces {
                if let Some(a) = &face.flow_accum {
                    for &v in a {
                        max_a = max_a.max(v);
                    }
                }
            }
            let max_f = (max_a.max(1)) as f32;

            for face in &planet.faces {
                if let Some(a) = &face.flow_accum {
                    let f32map: Vec<f32> = a.iter().map(|&v| v as f32).collect();
                    let filename = format!("{}_flow_{}.png", name, face.id.short_name());
                    let path = output.join(filename);
                    export_face_scalar_png_f32(
                        res,
                        &f32map,
                        &path,
                        0.0,
                        max_f,
                        image::codecs::png::CompressionType::Default,
                        image::codecs::png::FilterType::Adaptive,
                    ).unwrap_or_else(|e| {
                        eprintln!("Error exporting flow map: {}", e);
                        std::process::exit(1);
                    });
                }
            }
            println!("  Exported flow accumulation maps: {}_flow_*.png", name);
        }

        if river_map {
            for face in &planet.faces {
                if let Some(m) = &face.river_mask {
                    let filename = format!("{}_rivers_{}.png", name, face.id.short_name());
                    let path = output.join(filename);
                    export_face_mask_png_u8(
                        res,
                        m,
                        &path,
                        image::codecs::png::CompressionType::Default,
                        image::codecs::png::FilterType::Adaptive,
                    ).unwrap_or_else(|e| {
                        eprintln!("Error exporting river map: {}", e);
                        std::process::exit(1);
                    });
                }
            }
            println!("  Exported river mask maps: {}_rivers_*.png", name);
        }

        if deposition_map {
            let mut min_v = f32::MAX;
            let mut max_v = f32::MIN;
            for face in &planet.faces {
                if let Some(d) = &face.deposition {
                    for &v in d {
                        min_v = min_v.min(v);
                        max_v = max_v.max(v);
                    }
                }
            }
            let min_v = if min_v.is_finite() { min_v } else { 0.0 };
            let max_v = if max_v.is_finite() && max_v > min_v { max_v } else { min_v + 1e-6 };

            for face in &planet.faces {
                if let Some(d) = &face.deposition {
                    let filename = format!("{}_deposition_{}.png", name, face.id.short_name());
                    let path = output.join(filename);
                    export_face_scalar_png_f32(
                        res,
                        d,
                        &path,
                        min_v,
                        max_v,
                        image::codecs::png::CompressionType::Default,
                        image::codecs::png::FilterType::Adaptive,
                    ).unwrap_or_else(|e| {
                        eprintln!("Error exporting deposition map: {}", e);
                        std::process::exit(1);
                    });
                }
            }
            println!("  Exported deposition maps: {}_deposition_*.png", name);
        }
    }

    // Export climate monthly maps if requested (streamed month-by-month).
    if export_climate_monthly {
        let res = planet.resolution();
        let per_face = (res * res) as usize;
        let total = per_face * 6;

        println!("\nExporting monthly climate maps (12 months)...");

        // Flatten heights and compute coast distance for the current terrain state.
        let heights: Vec<f32> = {
            let mut out = vec![0.0f32; total];
            for (i, face) in planet.faces.iter().enumerate() {
                out[i * per_face..(i + 1) * per_face].copy_from_slice(&face.heights);
            }
            out
        };
        let coast_km = compute_coast_distance_km(&planet, climate_cfg.sea_level);
        let pre = precompute_lat_lon(res, &planet.faces);

        for month_idx in 0..climate_cfg.months.max(1) {
            let month = compute_month(
                &climate_cfg,
                month_idx,
                planet.radius,
                &heights,
                &coast_km,
                &pre,
            );

            // Compute per-month global ranges for stable visualization.
            let mut tmin = f32::MAX;
            let mut tmax = f32::MIN;
            for &t in &month.temperature_c {
                tmin = tmin.min(t);
                tmax = tmax.max(t);
            }
            let tmin = if tmin.is_finite() { tmin } else { -60.0 };
            let tmax = if tmax.is_finite() && tmax > tmin { tmax } else { tmin + 1e-6 };

            let mut pmin = f32::MAX;
            let mut pmax = f32::MIN;
            for &p in &month.precipitation_mm {
                pmin = pmin.min(p);
                pmax = pmax.max(p);
            }
            let pmin = if pmin.is_finite() { pmin } else { 0.0 };
            let pmax = if pmax.is_finite() && pmax > pmin { pmax } else { pmin + 1e-6 };

            for (fi, face) in planet.faces.iter().enumerate() {
                let a = fi * per_face;
                let b = (fi + 1) * per_face;
                let m = (month_idx as u32) + 1;

                let t_name = format!("{}_temp_m{:02}_{}.png", name, m, face.id.short_name());
                let t_path = output.join(t_name);
                export_face_scalar_png_f32(
                    res,
                    &month.temperature_c[a..b],
                    &t_path,
                    tmin,
                    tmax,
                    image::codecs::png::CompressionType::Default,
                    image::codecs::png::FilterType::Adaptive,
                )
                .unwrap_or_else(|e| {
                    eprintln!("Error exporting temperature month {}: {}", m, e);
                    std::process::exit(1);
                });

                let p_name = format!("{}_precip_m{:02}_{}.png", name, m, face.id.short_name());
                let p_path = output.join(p_name);
                export_face_scalar_png_f32(
                    res,
                    &month.precipitation_mm[a..b],
                    &p_path,
                    pmin,
                    pmax,
                    image::codecs::png::CompressionType::Default,
                    image::codecs::png::FilterType::Adaptive,
                )
                .unwrap_or_else(|e| {
                    eprintln!("Error exporting precipitation month {}: {}", m, e);
                    std::process::exit(1);
                });
            }

            println!("  Exported climate month {:02}: temp + precip", (month_idx + 1));
        }
    }

    let export_time = export_start.elapsed();
    let total_time = start.elapsed();

    println!("Export completed in {:.2?}", export_time);
    println!("\nTotal time: {:.2?}", total_time);
    println!("Done!");
}

fn run_info(resolution: u32) {
    let pixels_per_face = (resolution as u64) * (resolution as u64);
    let total_pixels = pixels_per_face * 6;

    let bytes_heights = total_pixels * 4; // f32
    let bytes_plate_ids = total_pixels * 8; // usize
    let bytes_tectonic_uplift = total_pixels * 4; // f32
    let bytes_png = pixels_per_face * 2 * 6; // 16-bit per face
    let bytes_raw_r16 = pixels_per_face * 2 * 6;
    let bytes_raw_r32 = pixels_per_face * 4 * 6;

    println!("Worldgen - Planet Configuration Info");
    println!("=====================================");
    println!();
    println!("Resolution: {}x{} per face", resolution, resolution);
    println!("Total faces: 6");
    println!();
    println!("Pixel counts:");
    println!("  Per face:  {:>12} pixels", pixels_per_face);
    println!("  Total:     {:>12} pixels", total_pixels);
    println!();
    println!("Memory usage (in-memory):");
    println!("  Heights:         {:>12} bytes ({:.2} MB)", bytes_heights, bytes_heights as f64 / 1024.0 / 1024.0);
    println!("  Plate IDs:       {:>12} bytes ({:.2} MB)", bytes_plate_ids, bytes_plate_ids as f64 / 1024.0 / 1024.0);
    println!("  Tectonic uplift: {:>12} bytes ({:.2} MB)", bytes_tectonic_uplift, bytes_tectonic_uplift as f64 / 1024.0 / 1024.0);
    let total_memory = bytes_heights + bytes_plate_ids + bytes_tectonic_uplift;
    println!("  Total:           {:>12} bytes ({:.2} MB)", total_memory, total_memory as f64 / 1024.0 / 1024.0);
    println!();
    println!("Export file sizes:");
    println!("  PNG (16-bit):   {:>8} bytes ({:.2} MB) - 6 files", bytes_png, bytes_png as f64 / 1024.0 / 1024.0);
    println!("  RAW (R16):      {:>8} bytes ({:.2} MB) - 6 files", bytes_raw_r16, bytes_raw_r16 as f64 / 1024.0 / 1024.0);
    println!("  RAW (R32):      {:>8} bytes ({:.2} MB) - 6 files", bytes_raw_r32, bytes_raw_r32 as f64 / 1024.0 / 1024.0);
    println!();

    // Engine compatibility notes
    println!("Engine compatibility notes:");
    if is_power_of_two(resolution) {
        println!("  Unity:    OK (power of 2)");
        println!("  Godot:    OK (power of 2)");
    } else {
        println!("  Unity:    May require power-of-2 resolution");
        println!("  Godot:    May require power-of-2 resolution");
    }

    let unreal_res = resolution + 1;
    if is_power_of_two(unreal_res - 1) {
        println!("  Unreal:   Recommended resolution {} (power-of-2 + 1)", unreal_res);
    } else {
        println!("  Unreal:   Current resolution OK");
    }
}

fn is_power_of_two(n: u32) -> bool {
    n > 0 && (n & (n - 1)) == 0
}
