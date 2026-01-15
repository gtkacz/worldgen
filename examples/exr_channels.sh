#!/usr/bin/env bash
set -euo pipefail

# OpenEXR multi-channel export (bash)

cargo run --release -- generate \
  --resolution 512 \
  --format exr \
  --output ./output_exr \
  --name planet \
  --earth-like \
  --exr-channels stable-schema

echo "Done. Outputs written to ./output_exr"

