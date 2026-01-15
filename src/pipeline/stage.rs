//! Generation stage trait and pipeline orchestration.

use std::collections::HashMap;
use thiserror::Error;
use rayon::prelude::*;
use crate::terrain::Planet;
use crate::noise::FractalNoiseConfig;
use crate::geometry::{face_uv_to_cube, spherify_point};
use crate::erosion::ErosionConfig;
use crate::erosion::ErosionBackend;
use crate::erosion::cpu::priority_flood::priority_flood_fill;
use crate::erosion::cpu::hydraulic::run_hydraulic_thermal_cpu;
use crate::erosion::cpu::rivers::{rivers_from_heights, write_river_outputs_to_planet};
use crate::erosion::wgpu::ErosionGpuContext;
use crate::erosion::wgpu::ErosionGpu;
use crate::climate::{ClimateConfig, compute_coast_distance_km, precompute_lat_lon, compute_month};
use crate::biomes::{BiomeConfig, compute_biomes};
use crate::tectonics::{
    TectonicConfig, TectonicPlate, SphericalVoronoi, PlateBoundary,
    detect_boundaries,
    detect_subduction, subduction_uplift, distance_to_trench,
    detect_collision, distance_to_suture, is_prowedge_side, himalayan_profile,
    detect_ridge, total_ridge_depth, distance_to_ridge,
};

/// Unique identifier for generation stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StageId {
    /// Initial heightmap generation from noise.
    Heightmap,
    /// Tectonic plate simulation (Phase 2).
    Tectonics,
    /// Hydraulic and thermal erosion (Phase 3).
    Erosion,
    /// Climate simulation (Phase 4).
    Climate,
    /// Biome classification (Phase 5).
    Biomes,
}

impl StageId {
    /// Returns the name of the stage.
    pub fn name(&self) -> &'static str {
        match self {
            StageId::Heightmap => "heightmap",
            StageId::Tectonics => "tectonics",
            StageId::Erosion => "erosion",
            StageId::Climate => "climate",
            StageId::Biomes => "biomes",
        }
    }
}

/// Configuration passed to each generation stage.
#[derive(Debug, Clone)]
pub struct StageConfig {
    /// Noise configuration for terrain generation.
    pub noise: FractalNoiseConfig,
    /// Additional stage-specific parameters.
    pub params: HashMap<String, f32>,
}

impl Default for StageConfig {
    fn default() -> Self {
        Self {
            noise: FractalNoiseConfig::default(),
            params: HashMap::new(),
        }
    }
}

impl StageConfig {
    /// Creates a new configuration with the given noise settings.
    pub fn with_noise(noise: FractalNoiseConfig) -> Self {
        Self {
            noise,
            params: HashMap::new(),
        }
    }

    /// Sets a stage parameter.
    pub fn set_param(&mut self, key: &str, value: f32) -> &mut Self {
        self.params.insert(key.to_string(), value);
        self
    }

    /// Gets a stage parameter, returning a default if not set.
    pub fn get_param(&self, key: &str, default: f32) -> f32 {
        self.params.get(key).copied().unwrap_or(default)
    }
}

/// Errors that can occur during pipeline execution.
#[derive(Error, Debug)]
pub enum PipelineError {
    #[error("Stage '{0}' failed: {1}")]
    StageFailed(String, String),
    #[error("Missing dependency: stage '{0}' requires '{1}'")]
    MissingDependency(String, String),
}

/// Trait for implementing generation stages.
///
/// Each stage transforms the planet data in some way, building upon
/// previous stages. The trait-based design allows for modular composition
/// and easy extension with new generation phases.
pub trait GenerationStage: Send + Sync {
    /// Returns the unique identifier for this stage.
    fn id(&self) -> StageId;

    /// Returns a human-readable name for the stage.
    fn name(&self) -> &str;

    /// Returns the stage IDs that must be executed before this stage.
    fn dependencies(&self) -> &[StageId] {
        &[]
    }

    /// Executes the generation stage, modifying the planet in place.
    ///
    /// # Arguments
    /// * `planet` - The planet to modify
    /// * `config` - Stage configuration parameters
    ///
    /// # Returns
    /// `Ok(())` on success, or an error describing what went wrong
    fn execute(&self, planet: &mut Planet, config: &StageConfig) -> Result<(), PipelineError>;

    /// Optional progress callback for long-running stages.
    ///
    /// # Arguments
    /// * `progress` - Value from 0.0 to 1.0 indicating completion
    fn on_progress(&self, _progress: f32) {
        // Default: do nothing
    }
}

/// Orchestrates multiple generation stages into a complete pipeline.
pub struct Pipeline {
    stages: Vec<Box<dyn GenerationStage>>,
    config: StageConfig,
}

impl Pipeline {
    /// Creates a new empty pipeline with the given configuration.
    pub fn new(config: StageConfig) -> Self {
        Self {
            stages: Vec::new(),
            config,
        }
    }

    /// Adds a stage to the pipeline.
    pub fn add_stage<S: GenerationStage + 'static>(&mut self, stage: S) -> &mut Self {
        self.stages.push(Box::new(stage));
        self
    }

    /// Returns the number of stages in the pipeline.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Executes all stages in order on the given planet.
    ///
    /// # Arguments
    /// * `planet` - The planet to generate
    ///
    /// # Returns
    /// `Ok(())` if all stages complete successfully
    pub fn run(&self, planet: &mut Planet) -> Result<(), PipelineError> {
        let mut completed: Vec<StageId> = Vec::new();

        for stage in &self.stages {
            // Check dependencies
            for dep in stage.dependencies() {
                if !completed.contains(dep) {
                    return Err(PipelineError::MissingDependency(
                        stage.name().to_string(),
                        dep.name().to_string(),
                    ));
                }
            }

            // Execute stage
            stage.execute(planet, &self.config)?;
            completed.push(stage.id());
        }

        Ok(())
    }

    /// Executes all stages with progress callbacks.
    ///
    /// # Arguments
    /// * `planet` - The planet to generate
    /// * `on_stage_start` - Called when each stage begins
    /// * `on_stage_complete` - Called when each stage finishes
    pub fn run_with_callbacks<F1, F2>(
        &self,
        planet: &mut Planet,
        mut on_stage_start: F1,
        mut on_stage_complete: F2,
    ) -> Result<(), PipelineError>
    where
        F1: FnMut(&str, usize, usize),
        F2: FnMut(&str, usize, usize),
    {
        let total = self.stages.len();
        let mut completed: Vec<StageId> = Vec::new();

        for (i, stage) in self.stages.iter().enumerate() {
            on_stage_start(stage.name(), i, total);

            // Check dependencies
            for dep in stage.dependencies() {
                if !completed.contains(dep) {
                    return Err(PipelineError::MissingDependency(
                        stage.name().to_string(),
                        dep.name().to_string(),
                    ));
                }
            }

            // Execute stage
            stage.execute(planet, &self.config)?;
            completed.push(stage.id());

            on_stage_complete(stage.name(), i, total);
        }

        Ok(())
    }
}

/// Heightmap generation stage using fractal noise.
pub struct HeightmapStage;

impl GenerationStage for HeightmapStage {
    fn id(&self) -> StageId {
        StageId::Heightmap
    }

    fn name(&self) -> &str {
        "Heightmap Generation"
    }

    fn execute(&self, planet: &mut Planet, config: &StageConfig) -> Result<(), PipelineError> {
        use crate::terrain::generate_heightmap;

        generate_heightmap(planet, &config.noise);
        Ok(())
    }
}

/// Tectonic plate simulation stage.
pub struct TectonicStage {
    pub config: TectonicConfig,
}

impl TectonicStage {
    /// Creates a new tectonic stage with the given configuration.
    pub fn new(config: TectonicConfig) -> Self {
        Self { config }
    }

    /// Creates a tectonic stage with Earth-like defaults.
    pub fn earth_like(seed: u64) -> Self {
        Self::new(TectonicConfig::earth_like(seed))
    }
}

impl GenerationStage for TectonicStage {
    fn id(&self) -> StageId {
        StageId::Tectonics
    }

    fn name(&self) -> &str {
        "Tectonic Simulation"
    }

    fn dependencies(&self) -> &[StageId] {
        &[StageId::Heightmap]
    }

    fn execute(&self, planet: &mut Planet, _config: &StageConfig) -> Result<(), PipelineError> {
        // Step 1: Generate Voronoi tessellation for plate boundaries
        let voronoi = SphericalVoronoi::new(
            self.config.num_plates,
            self.config.lloyd_iterations,
            self.config.seed,
        );

        // Step 2: Create plates with properties
        let plates = generate_plates(&voronoi, &self.config);

        // Step 3: Detect and classify boundaries
        let boundaries = detect_boundaries(&voronoi, &plates);

        // Step 4: Initialize tectonic data on all faces and assign plate IDs
        for face in &mut planet.faces {
            face.init_tectonic_data();
            assign_plate_ids(face, &voronoi);
        }

        // Step 5: Apply tectonic effects to each face
        for face in &mut planet.faces {
            apply_tectonic_effects(face, &boundaries, &plates, &self.config);
        }

        // Step 6: Apply tectonic uplift to height values
        for face in &mut planet.faces {
            face.apply_tectonic_uplift();
        }

        // Step 7: Store plate data in planet for later stages
        planet.plates = Some(plates);
        planet.boundaries = Some(boundaries);
        planet.voronoi = Some(voronoi);

        Ok(())
    }
}

/// Generates tectonic plates from Voronoi cell centers.
fn generate_plates(voronoi: &SphericalVoronoi, config: &TectonicConfig) -> Vec<TectonicPlate> {
    use rand::Rng;
    use rand_chacha::ChaCha8Rng;
    use rand::SeedableRng;

    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let areas = voronoi.estimate_cell_areas(500);

    // Sort plates by area to assign continental crust to larger plates
    let mut plate_indices: Vec<usize> = (0..voronoi.num_cells()).collect();
    plate_indices.sort_by(|&a, &b| areas[b].partial_cmp(&areas[a]).unwrap());

    // Determine how many plates should be continental
    let total_area: f32 = areas.iter().sum();
    let target_continental_area = total_area * config.continental_fraction;

    let mut continental_area = 0.0;
    let mut continental_plates = std::collections::HashSet::new();

    for &idx in &plate_indices {
        if continental_area < target_continental_area {
            continental_plates.insert(idx);
            continental_area += areas[idx];
        }
    }

    voronoi
        .cell_centers
        .iter()
        .enumerate()
        .map(|(id, &center)| {
            let is_continental = continental_plates.contains(&id);

            let mut plate = TectonicPlate::new(
                id,
                center,
                is_continental,
                config.velocity_scale * (1.0 + rng.random::<f32>() * config.velocity_randomness),
                config.seed.wrapping_add(id as u64 * 31),
            );

            plate.area = areas[id];
            plate
        })
        .collect()
}

/// Assigns plate IDs to each pixel in a face based on Voronoi tessellation.
fn assign_plate_ids(face: &mut crate::terrain::CubeFace, voronoi: &SphericalVoronoi) {
    let resolution = face.resolution;
    let face_id = face.id; // Copy face.id before closure

    // Process pixels in parallel for efficiency
    let plate_ids: Vec<usize> = (0..(resolution * resolution))
        .into_par_iter()
        .map(|idx| {
            let x = idx % resolution;
            let y = idx / resolution;
            let (u, v) = (
                (x as f32 + 0.5) / resolution as f32,
                (y as f32 + 0.5) / resolution as f32,
            );
            let cube_pos = face_uv_to_cube(face_id, u, v);
            let sphere_pos = spherify_point(cube_pos);
            voronoi.assign_point(sphere_pos)
        })
        .collect();

    face.plate_ids = Some(plate_ids);
}

/// Applies tectonic effects (subduction, collision, ridges) to a face.
fn apply_tectonic_effects(
    face: &mut crate::terrain::CubeFace,
    boundaries: &[PlateBoundary],
    plates: &[TectonicPlate],
    config: &TectonicConfig,
) {
    let resolution = face.resolution;
    let face_id = face.id; // Copy face.id before closure

    // Pre-compute tectonic zones for this face
    let subduction_zones: Vec<_> = boundaries
        .iter()
        .filter_map(|b| detect_subduction(b, plates))
        .collect();

    let collision_zones: Vec<_> = boundaries
        .iter()
        .filter_map(|b| detect_collision(b, plates))
        .collect();

    let ridges: Vec<_> = boundaries
        .iter()
        .filter_map(|b| detect_ridge(b, plates))
        .collect();

    // Calculate uplift for each pixel using parallel iteration
    let uplift: Vec<f32> = (0..(resolution * resolution))
        .into_par_iter()
        .map(|idx| {
            let x = idx % resolution;
            let y = idx / resolution;
            let (u, v) = (
                (x as f32 + 0.5) / resolution as f32,
                (y as f32 + 0.5) / resolution as f32,
            );
            let cube_pos = face_uv_to_cube(face_id, u, v);
            let sphere_pos = spherify_point(cube_pos);

            let mut total_uplift = 0.0;

            // Apply subduction effects
            for zone in &subduction_zones {
                let dist = distance_to_trench(sphere_pos, zone, plates);
                if dist.abs() < 500.0 {
                    let up = subduction_uplift(dist, zone.subduction_rate, zone.subduction_angle);
                    total_uplift += up * config.subduction_uplift_scale;
                }
            }

            // Apply collision effects
            for zone in &collision_zones {
                let dist = distance_to_suture(sphere_pos, zone, plates);
                if dist.abs() < 1000.0 {
                    let prowedge = is_prowedge_side(sphere_pos, zone, plates);
                    let up = himalayan_profile(dist, prowedge, zone.convergence_rate, 50.0);
                    total_uplift += up * config.collision_uplift_scale;
                }
            }

            // Apply ridge effects
            for ridge in &ridges {
                let dist = distance_to_ridge(sphere_pos, ridge);
                if dist < 1000.0 {
                    // Ridge creates relative elevation (less depression than surrounding seafloor)
                    let depth = total_ridge_depth(dist, ridge.spreading_rate);
                    // Convert ridge depth profile to uplift relative to average ocean depth
                    let avg_ocean_depth = -4.0;
                    let uplift_from_ridge = depth - avg_ocean_depth;
                    total_uplift += uplift_from_ridge * 0.5; // Scale factor
                }
            }

            total_uplift
        })
        .collect();

    face.tectonic_uplift = Some(uplift);
}

/// Erosion stage (Phase 3): hydraulic + thermal erosion, depression filling, river extraction.
pub struct ErosionStage {
    pub config: ErosionConfig,
}

impl ErosionStage {
    pub fn new(config: ErosionConfig) -> Self {
        Self { config }
    }
}

impl GenerationStage for ErosionStage {
    fn id(&self) -> StageId {
        StageId::Erosion
    }

    fn name(&self) -> &str {
        "Erosion Pipeline"
    }

    fn dependencies(&self) -> &[StageId] {
        // Erosion can operate on noise-only heightmaps (Phase 1) or tectonic heightmaps (Phase 2).
        &[StageId::Heightmap]
    }

    fn execute(&self, planet: &mut Planet, _config: &StageConfig) -> Result<(), PipelineError> {
        let res = planet.resolution();
        let per_face = (res * res) as usize;
        let total = per_face * 6;

        // Flatten current heights for CPU backend and deposition comparisons.
        let heights0: Vec<f32> = {
            let mut out = vec![0.0f32; total];
            for (i, face) in planet.faces.iter().enumerate() {
                out[i * per_face..(i + 1) * per_face].copy_from_slice(&face.heights);
            }
            out
        };

        #[derive(Debug)]
        struct ErosionOutputs {
            heights: Vec<f32>,
            water: Vec<f32>,
            sediment: Vec<f32>,
            deposition: Vec<f32>,
        }

        let mut run_gpu = || -> Result<ErosionOutputs, PipelineError> {
            let ctx = pollster::block_on(ErosionGpuContext::new()).map_err(|e| {
                PipelineError::StageFailed("Erosion Pipeline".to_string(), e.to_string())
            })?;
            let gpu = ErosionGpu::new(ctx);
            let out = gpu.run_hydraulic(planet, &self.config).map_err(|e| {
                PipelineError::StageFailed("Erosion Pipeline".to_string(), e.to_string())
            })?;
            Ok(ErosionOutputs {
                heights: out.heights,
                water: out.water,
                sediment: out.sediment,
                deposition: out.deposition,
            })
        };

        let run_cpu = || -> ErosionOutputs {
            let out = run_hydraulic_thermal_cpu(res, &heights0, &self.config);
            ErosionOutputs {
                heights: out.heights,
                water: out.water,
                sediment: out.sediment,
                deposition: out.deposition,
            }
        };

        let outputs = match self.config.backend {
            ErosionBackend::CpuOnly => run_cpu(),
            ErosionBackend::GpuOnly => run_gpu()?,
            ErosionBackend::Auto => match run_gpu() {
                Ok(o) => o,
                Err(_gpu_err) => run_cpu(),
            },
        };

        // Optionally keep intermediates on faces for export/inspection.
        if self.config.keep_intermediates {
            for (i, face) in planet.faces.iter_mut().enumerate() {
                face.water = Some(outputs.water[i * per_face..(i + 1) * per_face].to_vec());
                face.sediment = Some(outputs.sediment[i * per_face..(i + 1) * per_face].to_vec());
            }
        }

        // Store net deposition/erosion if requested (independent of keep_intermediates).
        if self.config.track_deposition {
            for (i, face) in planet.faces.iter_mut().enumerate() {
                face.deposition = Some(outputs.deposition[i * per_face..(i + 1) * per_face].to_vec());
            }
        }

        // Depression filling (Priority-Flood) using configured outlets.
        let filled = priority_flood_fill(res, &outputs.heights, self.config.outlet_model);

        // Write filled heights back to planet.
        {
            for (i, face) in planet.faces.iter_mut().enumerate() {
                face.heights.copy_from_slice(&filled[i * per_face..(i + 1) * per_face]);
            }
        }

        // River network extraction from filled heights.
        let (_down, accum, mask) = rivers_from_heights(res, &filled, self.config.river_accum_threshold);
        write_river_outputs_to_planet(planet, &accum, &mask);

        Ok(())
    }
}

/// Climate simulation stage (Phase 4): coast distance + annual climate summaries.
pub struct ClimateStage {
    pub config: ClimateConfig,
}

impl ClimateStage {
    pub fn new(config: ClimateConfig) -> Self {
        Self { config }
    }
}

impl GenerationStage for ClimateStage {
    fn id(&self) -> StageId {
        StageId::Climate
    }

    fn name(&self) -> &str {
        "Climate Simulation"
    }

    fn dependencies(&self) -> &[StageId] {
        // Climate can run after noise heightmap (and benefits from tectonics/erosion if present).
        &[StageId::Heightmap]
    }

    fn execute(&self, planet: &mut Planet, _config: &StageConfig) -> Result<(), PipelineError> {
        let res = planet.resolution();
        let per_face = (res * res) as usize;
        let total = per_face * 6;
        let months = self.config.months.max(1) as f32;

        // Flatten heights.
        let heights: Vec<f32> = {
            let mut out = vec![0.0f32; total];
            for (i, face) in planet.faces.iter().enumerate() {
                out[i * per_face..(i + 1) * per_face].copy_from_slice(&face.heights);
            }
            out
        };

        // Coast distance (km) across globe.
        let coast_km = compute_coast_distance_km(planet, self.config.sea_level);

        // Precompute sphere points + latitudes.
        let pre = precompute_lat_lon(res, &planet.faces);

        // Annual summaries.
        let mut temp_sum = vec![0.0f32; total];
        let mut temp_min = vec![f32::INFINITY; total];
        let mut temp_max = vec![f32::NEG_INFINITY; total];
        let mut precip_annual = vec![0.0f32; total];

        for m in 0..self.config.months.max(1) {
            let month = compute_month(
                &self.config,
                m,
                planet.radius,
                &heights,
                &coast_km,
                &pre,
            );

            for i in 0..total {
                let t = month.temperature_c[i];
                temp_sum[i] += t;
                temp_min[i] = temp_min[i].min(t);
                temp_max[i] = temp_max[i].max(t);
                precip_annual[i] += month.precipitation_mm[i];
            }
        }

        let temp_mean: Vec<f32> = temp_sum.into_iter().map(|s| s / months).collect();

        // Write outputs back to faces.
        for (fi, face) in planet.faces.iter_mut().enumerate() {
            let a = fi * per_face;
            let b = (fi + 1) * per_face;
            face.coast_distance_km = Some(coast_km[a..b].to_vec());
            face.temperature_mean_c = Some(temp_mean[a..b].to_vec());
            face.precip_annual_mm = Some(precip_annual[a..b].to_vec());
            face.temp_min_month_c = Some(temp_min[a..b].to_vec());
            face.temp_max_month_c = Some(temp_max[a..b].to_vec());
        }

        Ok(())
    }
}

/// Biome generation stage (Phase 5): biome IDs + derived maps.
pub struct BiomeStage {
    pub config: BiomeConfig,
}

impl BiomeStage {
    pub fn new(config: BiomeConfig) -> Self {
        Self { config }
    }
}

impl GenerationStage for BiomeStage {
    fn id(&self) -> StageId {
        StageId::Biomes
    }

    fn name(&self) -> &str {
        "Biome Generation"
    }

    fn dependencies(&self) -> &[StageId] {
        &[StageId::Climate]
    }

    fn execute(&self, planet: &mut Planet, _config: &StageConfig) -> Result<(), PipelineError> {
        let res = planet.resolution();
        let per_face = (res * res) as usize;
        let total = per_face * 6;

        // Validate required climate fields exist.
        for face in &planet.faces {
            if face.temperature_mean_c.is_none()
                || face.precip_annual_mm.is_none()
                || face.temp_min_month_c.is_none()
                || face.temp_max_month_c.is_none()
            {
                return Err(PipelineError::StageFailed(
                    self.name().to_string(),
                    "Missing climate data on CubeFace (run Climate stage first)".to_string(),
                ));
            }
        }

        // Flatten inputs (heights + climate).
        let heights: Vec<f32> = {
            let mut out = vec![0.0f32; total];
            for (i, face) in planet.faces.iter().enumerate() {
                out[i * per_face..(i + 1) * per_face].copy_from_slice(&face.heights);
            }
            out
        };

        let temp_mean: Vec<f32> = {
            let mut out = vec![0.0f32; total];
            for (i, face) in planet.faces.iter().enumerate() {
                out[i * per_face..(i + 1) * per_face]
                    .copy_from_slice(face.temperature_mean_c.as_ref().unwrap());
            }
            out
        };

        let precip_annual: Vec<f32> = {
            let mut out = vec![0.0f32; total];
            for (i, face) in planet.faces.iter().enumerate() {
                out[i * per_face..(i + 1) * per_face]
                    .copy_from_slice(face.precip_annual_mm.as_ref().unwrap());
            }
            out
        };

        let temp_min_month: Vec<f32> = {
            let mut out = vec![0.0f32; total];
            for (i, face) in planet.faces.iter().enumerate() {
                out[i * per_face..(i + 1) * per_face]
                    .copy_from_slice(face.temp_min_month_c.as_ref().unwrap());
            }
            out
        };

        // Optional river mask (may be absent if erosion is skipped).
        let river_mask_flat: Option<Vec<u8>> = {
            let any = planet.faces.iter().any(|f| f.river_mask.is_some());
            if !any {
                None
            } else {
                let mut out = vec![0u8; total];
                for (i, face) in planet.faces.iter().enumerate() {
                    if let Some(m) = &face.river_mask {
                        out[i * per_face..(i + 1) * per_face].copy_from_slice(m);
                    }
                }
                Some(out)
            }
        };

        let outputs = compute_biomes(
            res,
            &self.config,
            &heights,
            &temp_mean,
            &precip_annual,
            &temp_min_month,
            river_mask_flat.as_deref(),
        );

        // Write outputs back to faces.
        for (fi, face) in planet.faces.iter_mut().enumerate() {
            let a = fi * per_face;
            let b = (fi + 1) * per_face;
            face.land_mask = Some(outputs.land_mask[a..b].to_vec());
            face.biome_ids = Some(outputs.biome_ids[a..b].to_vec());
            face.roughness = Some(outputs.roughness[a..b].to_vec());
            face.albedo = Some(outputs.albedo[a..b].to_vec());
            face.vegetation_density = Some(outputs.vegetation_density[a..b].to_vec());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::erosion::ErosionBackend;
    use crate::biomes::BiomeConfig;

    #[test]
    fn test_stage_config() {
        let mut config = StageConfig::default();
        config.set_param("erosion_strength", 0.5);

        assert_eq!(config.get_param("erosion_strength", 0.0), 0.5);
        assert_eq!(config.get_param("missing", 1.0), 1.0);
    }

    #[test]
    fn test_pipeline_execution() {
        let config = StageConfig::default();
        let mut pipeline = Pipeline::new(config);
        pipeline.add_stage(HeightmapStage);

        let mut planet = Planet::new(32, 42, 6371.0);
        pipeline.run(&mut planet).unwrap();

        // Verify heights were generated
        let (min, max) = planet.height_range();
        assert!(min < max, "Heightmap should have variation");
    }

    #[test]
    fn test_pipeline_cpu_erosion_stage_produces_rivers_and_deposition() {
        let config = StageConfig::default();
        let mut pipeline = Pipeline::new(config);
        pipeline.add_stage(HeightmapStage);

        let erosion_cfg = ErosionConfig {
            backend: ErosionBackend::CpuOnly,
            hydraulic_steps: 25,
            rainfall: 0.01,
            evaporation: 0.02,
            thermal_iterations: 10,
            river_accum_threshold: 25, // low threshold so mask isn't trivially empty
            keep_intermediates: false,
            track_deposition: true,
            ..Default::default()
        };
        pipeline.add_stage(ErosionStage::new(erosion_cfg));

        let mut planet = Planet::new(32, 42, 6371.0);
        pipeline.run(&mut planet).unwrap();

        for face in &planet.faces {
            assert!(face.flow_accum.is_some(), "flow_accum should be populated");
            assert!(face.river_mask.is_some(), "river_mask should be populated");
            assert!(face.deposition.is_some(), "deposition should be populated when enabled");
        }
    }

    #[test]
    fn test_pipeline_with_callbacks() {
        let config = StageConfig::default();
        let mut pipeline = Pipeline::new(config);
        pipeline.add_stage(HeightmapStage);

        let mut planet = Planet::new(16, 42, 6371.0);
        let mut started = false;
        let mut completed = false;

        pipeline
            .run_with_callbacks(
                &mut planet,
                |name, _, _| {
                    assert_eq!(name, "Heightmap Generation");
                    started = true;
                },
                |name, _, _| {
                    assert_eq!(name, "Heightmap Generation");
                    completed = true;
                },
            )
            .unwrap();

        assert!(started);
        assert!(completed);
    }

    #[test]
    fn test_stage_id_name() {
        assert_eq!(StageId::Heightmap.name(), "heightmap");
        assert_eq!(StageId::Tectonics.name(), "tectonics");
    }

    #[test]
    fn test_pipeline_climate_then_biomes_populates_outputs() {
        let config = StageConfig::default();
        let mut pipeline = Pipeline::new(config);
        pipeline.add_stage(HeightmapStage);
        pipeline.add_stage(ClimateStage::new(ClimateConfig::earth_like()));
        pipeline.add_stage(BiomeStage::new(BiomeConfig { sea_level: 0.0, seed: 42, ..Default::default() }));

        let mut planet = Planet::new(32, 42, 6371.0);
        pipeline.run(&mut planet).unwrap();

        for face in &planet.faces {
            assert!(face.land_mask.is_some(), "land_mask should be populated");
            assert!(face.biome_ids.is_some(), "biome_ids should be populated");
            assert!(face.roughness.is_some(), "roughness should be populated");
            assert!(face.albedo.is_some(), "albedo should be populated");
            assert!(face.vegetation_density.is_some(), "vegetation_density should be populated");

            let land = face.land_mask.as_ref().unwrap();
            let biomes = face.biome_ids.as_ref().unwrap();
            assert_eq!(land.len(), biomes.len());
            // Masking invariant: ocean pixels are biome_id 0.
            for i in 0..land.len() {
                if land[i] == 0 {
                    assert_eq!(biomes[i], 0);
                }
            }
        }
    }
}
