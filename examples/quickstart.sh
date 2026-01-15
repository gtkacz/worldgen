#!/usr/bin/env bash
set -euo pipefail

# Quickstart (bash)

cargo run --release -- generate \
  --resolution 512 \
  --format png \
  --output ./output \
  --name planet \
  --earth-like \
  --plate-map \
  --boundary-map \
  --water-map \
  --flow-map \
  --river-map \
  --biome-map \
  --normal-map \
  --normal-strength 2.0

echo "Done. Outputs written to ./output"

