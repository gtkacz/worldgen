# OpenEXR multi-channel export (PowerShell)

$ErrorActionPreference = "Stop"

# Generate full pipeline and export EXR.
# Note: `--exr-channels stable-schema` ensures consistent channel names even if you skip stages.
cargo run --release -- generate `
  --resolution 512 `
  --format exr `
  --output ./output_exr `
  --name planet `
  --earth-like `
  --exr-channels stable-schema

Write-Host "Done. Outputs written to ./output_exr"

