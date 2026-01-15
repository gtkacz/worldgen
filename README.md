# worldgen

Procedural Earth-like planet generator in Rust using cube-sphere geometry, tectonics, erosion, climate, and biome classification.

## Build

Requirements:
- Rust (stable toolchain)

```bash
cargo build --release
```

## Run (CLI)

The main entrypoint is:

```bash
cargo run --release -- generate --help
```

### Basic heightmaps (PNG)

```bash
cargo run --release -- generate --resolution 512 --format png --output ./output --name planet
```

This writes 6 files:
- `planet_posx.png`, `planet_negx.png`, `planet_posy.png`, `planet_negy.png`, `planet_posz.png`, `planet_negz.png`

### Assembled equirectangular maps (PNG)

Export additional assembled equirectangular (lat/lon) PNGs:

```bash
cargo run --release -- generate --resolution 512 --format png --output ./output --name planet --equirect
```

Writes:
- `planet_equirect_height.png` (16-bit grayscale)
- `planet_equirect_biomes.png` (RGB, only if biome data exists)

### OpenEXR multi-channel export

```bash
cargo run --release -- generate --resolution 512 --format exr --output ./output --name planet --exr-channels all-available
```

Writes 6 files:
- `planet_posx.exr` ... `planet_negz.exr`

`--exr-channels` presets:
- `height-only`: only `height`
- `all-available`: `height` + any other channels currently populated on that face
- `stable-schema`: always writes the full schema, filling missing channels with 0.0

#### EXR channel names

All channels are stored as **32-bit float** samples with these deterministic names:

- Always:
  - `height`
- Tectonics (if present / or filled in stable schema):
  - `plate_id` (float-encoded)
  - `tectonic_uplift`
- Erosion (if present):
  - `water`
  - `sediment`
  - `deposition`
  - `flow_accum` (float-cast)
  - `river_mask` (0..1)
- Climate annual summaries (if present):
  - `coast_km`
  - `temp_mean_c`
  - `temp_min_c`
  - `temp_max_c`
  - `precip_annual_mm`
- Biomes (if present):
  - `land_mask` (0..1)
  - `biome_id` (float-encoded)
  - `roughness`
  - `albedo`
  - `vegetation`

### Normal maps

Generate per-face normal maps (Sobel over the final heightfield):

```bash
cargo run --release -- generate --resolution 512 --format png --output ./output --name planet --normal-map --normal-strength 2.0
```

Writes:
- `planet_normal_posx.png` ... `planet_normal_negz.png`

### Optional debug/inspection exports

The CLI supports many optional exports (plates, boundaries, water/sediment/flow, climate months, biome previews, etc.). See:

```bash
cargo run --release -- generate --help
```

## Blender quick preview

There is a demo importer script at:
- `tools/blender_import_worldgen.py`

Workflow:
1) Generate PNG outputs (e.g. `--biome-map` for biome preview, or `--plate-map`, or `--boundary-map`).\n+2) In Blender → Scripting workspace → open `tools/blender_import_worldgen.py`.\n+3) Edit `OUTPUT_DIR`, `BASE_NAME`, and `MAP_KIND` at the top of the script.\n+4) Run Script.\n+
The script cube-map-samples the 6 face images onto an Ico Sphere material for quick visualization.

## Interactive viewer (windowed)

Build/run the viewer binary. It loads the 6 face PNGs and renders an interactive assembled equirect view.

```bash
cargo run --release --bin worldgen_viewer -- --input ./output --name planet
```

If you also exported biome preview faces (`--biome-map`), the viewer will auto-detect them at `{name}_biomes_*.png`.
Use `1` for height and `2` for biomes. Drag to pan; mouse wheel zooms.
