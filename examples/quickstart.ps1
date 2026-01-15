# Quickstart (PowerShell)

$ErrorActionPreference = "Stop"

# Basic heightmaps + biome preview + normal maps.
cargo run --release -- generate `
  --resolution 512 `
  --format png `
  --output ./output `
  --name planet `
  --earth-like `
  --plate-map `
  --boundary-map `
  --water-map `
  --flow-map `
  --river-map `
  --biome-map `
  --normal-map `
  --normal-strength 2.0

Write-Host "Done. Outputs written to ./output"

