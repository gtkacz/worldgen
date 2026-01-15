# Procedural Earth-Like Planet Generator: Rust Implementation Plan

A complete Earth-like exoplanet generator in Rust requires integrating tectonic simulation, multi-scale erosion, climate modeling, and biome classification into a unified pipeline. This plan details specific algorithms, Rust crates with versions, paper references, and a phased roadmap from basic terrain to full climate simulation—architected for whole-planet scale with extensibility to higher detail levels.

## Core technology stack and recommended crates

The Rust ecosystem provides excellent tools for this project, with **simdnoise** delivering 8-30× performance gains over scalar noise through AVX2 vectorization, and **rayon** enabling trivial parallelization across CPU cores.

| Component | Primary Crate | Version | Fallback |
|-----------|--------------|---------|----------|
| Noise generation | simdnoise | 3.1.4+ | noise-rs 0.9.0 |
| Linear algebra | glam | 0.30.9 | nalgebra 0.33+ |
| Image export | image | 0.25+ | exr 1.74+ |
| CPU parallelism | rayon | 1.11.0 | — |
| GPU compute | wgpu | 24+ | — |
| Sphere mesh | hexasphere | 15.0 | genmesh |
| Serialization | bincode + serde | 1.0.x | postcard |
| Memory mapping | memmap2 | 0.9+ | — |

**simdnoise** supports 4D Simplex noise essential for seamless spherical mapping—sampling 3D positions on the sphere's surface eliminates polar distortion artifacts that plague 2D UV-based approaches. The **glam** crate outperforms alternatives in graphics workloads with SSE2/NEON SIMD, while **hexasphere** generates high-quality icosphere meshes with geometric slerp interpolation.

```toml
[dependencies]
simdnoise = "3.1"
noise = "0.9"
glam = { version = "0.30", features = ["bytemuck", "serde"] }
image = { version = "0.25", features = ["png", "tiff", "exr"] }
exr = "1.74"
rayon = "1.11"
hexasphere = "15"
bytemuck = { version = "1.14", features = ["derive"] }
serde = { version = "1.0", features = ["derive"] }
bincode = "1.3"
memmap2 = "0.9"
```

## Cube-sphere geometry provides the optimal foundation

After evaluating icospheres, HEALPix, and geodesic grids, the **spherified cube-sphere** emerges as the best choice for procedural planets. Each of six cube faces becomes a quadtree root enabling natural LOD subdivision, while cubemap textures enjoy native GPU support across all graphics APIs.

The spherification formula transforms cube coordinates to sphere with better uniformity than simple normalization:

```rust
fn spherify_point(cube_pos: Vec3) -> Vec3 {
    let x2 = cube_pos.x * cube_pos.x;
    let y2 = cube_pos.y * cube_pos.y;
    let z2 = cube_pos.z * cube_pos.z;
    
    Vec3::new(
        cube_pos.x * (1.0 - y2/2.0 - z2/2.0 + y2*z2/3.0).sqrt(),
        cube_pos.y * (1.0 - x2/2.0 - z2/2.0 + x2*z2/3.0).sqrt(),
        cube_pos.z * (1.0 - x2/2.0 - y2/2.0 + x2*y2/3.0).sqrt(),
    )
}
```

Memory requirements scale predictably: a **2048×2048** per-face resolution yields **25.2 million vertices** consuming approximately **300 MB** for positions alone, or **~960 MB** with full vertex data. The quadtree LOD system reduces active memory by loading only nearby chunks at full resolution.

For future LOD expansion, implement **CDLOD** (Continuous Distance-Dependent LOD) from Filip Strugar's 2009 paper—this approach uses GPU tessellation with smooth geomorphing between levels, eliminating visible "popping" artifacts and T-junction cracks without requiring mesh skirts.

## Tectonic plate simulation creates continental structure

Procedural plate tectonics follows a three-stage pipeline: boundary generation, movement simulation, and feature formation. The foundational paper is **Cortial et al. (2019) "Procedural Tectonic Planets"** from Computer Graphics Forum.

### Generating plate boundaries with Voronoi tessellation

Distribute **12-40 seed points** on the sphere using Fibonacci spiral sampling for near-uniform distribution, then construct a spherical Voronoi tessellation. Apply Lloyd's relaxation (3-5 iterations) to regularize cell sizes, followed by boundary noise warping for organic fracture lines.

```rust
struct TectonicPlate {
    id: u16,
    crust_type: CrustType,      // Continental or Oceanic
    thickness: f32,              // 5-70 km (oceanic thinner)
    density: f32,                // 2.7-3.0 g/cm³
    velocity: Vec3,              // Angular velocity vector
    age: f32,                    // Millions of years
}

enum CrustType {
    Continental { shield_age: f32 },  // Thicker, older, less dense
    Oceanic { ridge_distance: f32 },  // Thinner, younger, denser
}
```

### Simulating plate movement and collisions

Each plate moves as rigid body rotation around an axis through the planet center. Collision detection identifies converging boundaries where plates interpenetrate beyond a threshold (~300 km). The denser plate subducts beneath the lighter—oceanic crust (3.0 g/cm³) typically subducts under continental (2.7 g/cm³).

**Subduction uplift** follows the transfer function from Cortial et al.:
```
uplift(p) = u₀ × f(distance) × g(velocity) × h(elevation)
```
Where f(d) peaks at ~500 km from the subduction front and decays to zero at 1800 km, creating realistic volcanic arc profiles like the Andes.

**Collision orogeny** (Himalayan-type) applies discrete elevation surges with radial falloff:
```
Δz(p) = Δc × Area × (1 - (d/r)²)²
```
Collision radius r scales with collision velocity and plate area, producing wider mountain ranges from faster, larger collisions.

### Divergent boundaries and rifting

Oceanic ridges form at divergent boundaries—interpolate elevation between plates using a ridge profile template, reset crust age to zero, and record ridge orientation for later detail amplification with oriented Gabor noise. Plate rifting occurs probabilistically based on plate area and age, subdividing large plates via internal Voronoi tessellation.

## Erosion algorithms sculpt realistic landforms

Erosion transforms blocky tectonic features into natural terrain through two complementary processes: hydraulic (water) and thermal (weathering) erosion.

### Hydraulic erosion with the pipe model

The GPU-accelerated shallow water model from **Mei, Decaudin, and Hu (2007) "Fast Hydraulic Erosion Simulation on GPU"** provides the best balance of realism and performance. Each cell tracks water height, suspended sediment, and outflow flux to four neighbors.

**Five-step simulation per timestep:**

1. **Water increment**: Add rainfall based on climate simulation
2. **Flow simulation**: Calculate outflow using height differences with scaling to prevent negative water
3. **Erosion-deposition**: Compare sediment transport capacity C = Kc × sin(α) × |v| against current load
4. **Sediment transport**: Semi-Lagrangian advection along velocity field
5. **Evaporation**: Reduce water by evaporation factor Ke

**Key parameters** (tunable for different planetary conditions):
- Erosion rate Ks: 0.01-0.05
- Deposition rate Kd: 0.01-0.05
- Sediment capacity Kc: 0.01-0.1
- Evaporation Ke: 0.01-0.02

Performance benchmarks from 2007 hardware show **22.88 ms for 2048×2048** grids—modern GPUs via wgpu compute shaders will be significantly faster.

### Adapting erosion for spherical surfaces

Apply erosion per cube face with special boundary handling at face edges. Gradient calculations use spherical neighbor positions, and water flow follows great-circle arcs rather than planar vectors. The cube-sphere projection minimizes distortion compared to equirectangular approaches.

### Depression filling and drainage networks

**Priority-Flood** (Barnes, Lehman, Mulla 2014) fills terrain depressions in optimal O(n) time for integer data or O(n log n) for floats—critical for generating coherent river networks. The algorithm uses a priority queue seeded with boundary cells, propagating minimum elevations inward.

River extraction follows drainage area accumulation: compute D8 flow directions, accumulate upstream cell counts, and threshold for river designation. Lakes form at depressions where filled cells don't reach the boundary—their level equals the spill point elevation.

### Thermal erosion for talus slopes

Material moves when slope exceeds the angle of repose (30-45° depending on rock type). Each iteration redistributes a fraction of the height difference to lower neighbors, smoothing cliffs into natural scree slopes over **100-500 iterations**.

## Climate simulation determines biome placement

Climate systems use simplified models from atmospheric science—sufficient for believable results without full fluid dynamics simulation.

### Temperature distribution combines three factors

**Base temperature** follows a latitude gradient modified by altitude:
```rust
fn calculate_temperature(latitude: f32, altitude: f32) -> f32 {
    let lat_factor = 30.0 * latitude.to_radians().cos();
    let alt_factor = 6.5 * (altitude / 1000.0);  // 6.5°C/km lapse rate
    lat_factor - alt_factor - 10.0  // Offset for Earth-like range
}
```

**Maritime buffering** reduces seasonal extremes within 300-500 km of coastlines by blending with ocean temperatures. Continental interiors experience harsher swings (hot summers, frigid winters).

### Wind and precipitation patterns

The **three-cell model** (Hadley, Ferrel, Polar) creates predictable wind bands: easterlies 0-30°, westerlies 30-60°, polar easterlies 60-90°. The Intertropical Convergence Zone (ITCZ) oscillates ±10° seasonally following solar heating.

**Orographic precipitation** creates rain shadows—moisture-laden air rises over mountains, cools adiabatically to saturation, precipitates on windward slopes, then descends warm and dry on leeward sides. Implement as moisture transport along wind vectors with elevation-triggered precipitation:
```
P = P₀ × exp(-elevation/H_scale) × wind_perpendicular
```
Where H_scale ≈ 2-3 km controls precipitation sensitivity to altitude.

### Biome classification using the Whittaker diagram

The **Whittaker biome diagram** provides the simplest effective classification for procedural generation—a 2D lookup table with temperature and precipitation axes. This outperforms Köppen for games/visualization while maintaining scientific grounding.

| Biome | Temp Min (°C) | Temp Max | Precip Min (mm) | Precip Max |
|-------|---------------|----------|-----------------|------------|
| Tropical Rainforest | 25 | 30 | 2500 | 10000 |
| Savanna | 20 | 30 | 500 | 1500 |
| Subtropical Desert | 15 | 30 | 0 | 250 |
| Temperate Deciduous | 5 | 20 | 750 | 1500 |
| Taiga/Boreal | -5 | 5 | 400 | 1000 |
| Tundra | -10 | 0 | 150 | 500 |
| Ice Cap | — | -10 | — | — |

Apply **noise-based boundary distortion** for organic ecotones rather than sharp biome transitions. Altitude-adjusted biome placement uses the equivalence: each 1000m elevation ≈ 6° latitude increase.

## Export pipeline supports major 3D software

### Heightmap format recommendations

**16-bit PNG** provides universal compatibility across Blender, Unity, and Unreal, offering 65,536 elevation levels. For professional workflows, **OpenEXR** stores true floating-point heights in meters with multi-channel support—pack height, moisture, temperature, and biome ID into a single file.

**Critical engine requirements:**
- **Unreal Engine**: Dimensions must be power-of-2 + 1 (1009, 2017, 4033)—pure powers of 2 cause edge stretching
- **Unity**: RAW format with power-of-2 dimensions, specify byte order (little-endian on Windows)
- **Godot/Terrain3D**: OpenEXR or RAW with 16/32-bit float

### Normal map generation from heightmaps

The **Sobel operator** remains the standard algorithm:
```rust
fn generate_normal(heights: &[f32], x: usize, y: usize, width: usize, strength: f32) -> Vec3 {
    let sobel_x = heights[idx(x+1, y-1)] - heights[idx(x-1, y-1)]
                + 2.0 * heights[idx(x+1, y)] - 2.0 * heights[idx(x-1, y)]
                + heights[idx(x+1, y+1)] - heights[idx(x-1, y+1)];
    let sobel_y = heights[idx(x-1, y+1)] - heights[idx(x-1, y-1)]
                + 2.0 * heights[idx(x, y+1)] - 2.0 * heights[idx(x, y-1)]
                + heights[idx(x+1, y+1)] - heights[idx(x+1, y-1)];
    
    Vec3::new(-sobel_x, -sobel_y, 1.0 / strength).normalize()
}
```

Export tangent-space normals encoded as RGB (N.xy × 0.5 + 0.5, N.z) for terrain displacement.

### Additional exportable map types

- **Moisture/Precipitation**: 8-16 bit, values 0-1
- **Temperature**: 8-16 bit, encode with known min/max
- **Biome ID**: 8-bit indexed color with palette lookup
- **Plate boundaries**: Binary mask or distance field
- **Ocean depth**: Same format as heightmap, negative values
- **River mask**: Binary or flow accumulation values

## Architecture for modularity and future expansion

### Data structures for planetary representation

```rust
pub struct Planet {
    pub radius: f32,
    pub seed: u64,
    pub faces: [CubeFace; 6],
    pub plates: Vec<TectonicPlate>,
    pub climate: ClimateData,
}

pub struct CubeFace {
    pub resolution: u32,
    pub heights: Vec<f32>,           // Dense: every cell
    pub moisture: Vec<f32>,          // Dense: climate data
    pub temperature: Vec<f32>,       // Dense: climate data
    pub biome_ids: Vec<u8>,          // Dense: biome classification
    pub rivers: HashMap<u32, River>, // Sparse: ~1-5% of cells
}

pub struct QuadTree {
    pub root: QuadNode,
    pub max_depth: u8,
}

pub enum QuadNode {
    Leaf { tile_data: Box<TileData> },
    Branch { children: Box<[QuadNode; 4]> },
}
```

Use **dense storage** (Vec) for data present in every cell (heights, biomes) and **sparse storage** (HashMap) for rare features (rivers, cities, points of interest).

### Trait-based pipeline composition

```rust
pub trait GenerationStage {
    fn execute(&self, planet: &mut Planet, config: &StageConfig) -> Result<()>;
    fn dependencies(&self) -> &[StageId];
}

pub struct Pipeline {
    stages: Vec<Box<dyn GenerationStage>>,
}

impl Pipeline {
    pub fn run(&self, planet: &mut Planet) -> Result<()> {
        for stage in &self.stages {
            stage.execute(planet, &self.config)?;
        }
        Ok(())
    }
}
```

### Memory efficiency for large planets

Use **memmap2** for memory-mapped file access, enabling planets larger than available RAM:
```rust
use memmap2::MmapMut;

fn stream_heightmap(path: &Path) -> Result<MmapMut> {
    let file = OpenOptions::new().read(true).write(true).create(true).open(path)?;
    file.set_len(PLANET_SIZE_BYTES)?;
    unsafe { MmapMut::map_mut(&file) }
}
```

Implement an **LRU tile cache** for quadtree streaming—load tiles on demand, evict least-recently-used when memory pressure increases.

### Serialization with versioning

Combine human-readable headers with binary data for both flexibility and performance:
```rust
#[derive(Serialize, Deserialize)]
pub struct PlanetFile {
    pub version: u32,              // Always first for migration
    pub header: PlanetHeader,      // JSON-serializable metadata
    pub data_offset: u64,          // Pointer to binary section
}

// Binary data uses bincode for speed
pub struct PlanetBinaryData {
    pub heights: Vec<f32>,
    pub biomes: Vec<u8>,
    pub plate_boundaries: Vec<u32>,
}
```

Include explicit version numbers and maintain migration functions for backwards compatibility as the format evolves.

## Phased implementation roadmap

### Phase 1: Foundation (weeks 1-3)
- Implement cube-sphere coordinate system with spherification
- Integrate simdnoise for 3D/4D noise sampling on sphere
- Create basic heightmap generation with multi-octave fractal noise
- Build PNG and RAW export pipeline
- Establish test harness validating coordinate conversions

### Phase 2: Tectonic system (weeks 4-6)
- Implement spherical Voronoi for plate boundaries
- Add plate movement simulation with angular velocities
- Create collision detection and subduction modeling
- Generate mountain ranges from convergent boundaries
- Produce oceanic ridges at divergent zones

### Phase 3: Erosion pipeline (weeks 7-9)
- Port Mei et al. hydraulic erosion to wgpu compute shaders
- Implement Priority-Flood depression filling
- Add thermal erosion for slope smoothing
- Extract river networks from drainage accumulation
- Generate sediment deposits in basins

### Phase 4: Climate simulation (weeks 10-11)
- Calculate temperature from latitude, altitude, and coastline proximity
- Simulate wind patterns using three-cell model approximation
- Implement moisture transport with orographic precipitation
- Add seasonal variation based on axial tilt parameter

### Phase 5: Biome generation (week 12)
- Implement Whittaker diagram lookup with configurable thresholds
- Add ecotone blending with noise-perturbed boundaries
- Generate derived maps: roughness, albedo suggestions, vegetation density
- Create biome-colored preview renders

### Phase 6: Polish and export (weeks 13-14)
- Implement OpenEXR multi-channel export
- Add normal map generation with configurable strength
- Build Blender import script for demonstration
- Create comprehensive documentation and example configurations
- Performance profiling and optimization pass

## Key academic references

The algorithms above derive from peer-reviewed research providing reproducible implementations:

- **Cortial, Peytavie, Galin, Guérin (2019)**: "Procedural Tectonic Planets" — Computer Graphics Forum — Plate dynamics and orogeny
- **Mei, Decaudin, Hu (2007)**: "Fast Hydraulic Erosion Simulation on GPU" — Pacific Graphics — Pipe model erosion
- **Barnes, Lehman, Mulla (2014)**: "Priority-Flood: An Optimal Depression-Filling Algorithm" — Computers & Geosciences — Drainage networks
- **Strugar (2009)**: "Continuous Distance-Dependent LOD for Rendering Heightmaps" — GPU tessellation LOD
- **Musgrave, Kolb, Mace (1989)**: "The Synthesis and Rendering of Eroded Fractal Terrains" — SIGGRAPH — Foundational terrain synthesis

## Conclusion

This implementation plan provides a complete path from empty repository to Earth-like exoplanet generator. The cube-sphere geometry enables both efficient storage and natural LOD subdivision; the tectonic system creates geologically-plausible continental arrangements; erosion sculpts believable landforms; and climate simulation drives coherent biome placement. The modular trait-based architecture allows each system to evolve independently while the versioned serialization ensures long-term data compatibility.

Starting with Phase 1's foundational noise-based terrain allows rapid visual progress, while subsequent phases layer increasingly sophisticated simulation. The recommended crates—simdnoise, glam, rayon, wgpu—represent the current Rust ecosystem's best options for this domain, balancing performance with maintainability. By following this roadmap, a solo developer can build a production-quality planet generator within 14 weeks, with clear extension points for future detail enhancement.