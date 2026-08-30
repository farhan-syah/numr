#!/usr/bin/env bash
# Verify the packaged crate tarball carries the non-Rust sources numr's build
# needs.
#
#   scripts/ci/verify_package_contents.sh <path/to/numr-X.Y.Z.crate>
#
# `cargo publish --dry-run` verifies the packaged crate with DEFAULT FEATURES
# ONLY. numr's `build.rs` compiles CUDA kernels from `.cu`/`.cuh` sources, and
# that step never runs under default features — so a kernel source missing
# from the package publishes cleanly and then breaks every downstream
# `--features cuda` build. No GitHub runner has `nvcc`, so compiling is not an
# option here; checking the packaged file list is.

set -euo pipefail

CRATE="${1:?usage: verify_package_contents.sh <path/to/numr-X.Y.Z.crate>}"

test -f "$CRATE" || {
  echo "::error::crate tarball not found: $CRATE"
  exit 1
}

FILES=$(tar tzf "$CRATE")

require_any() {
  local label="$1"
  local pattern="$2"
  local count
  count=$(grep -cE "$pattern" <<<"$FILES" || true)
  if [[ "$count" -eq 0 ]]; then
    echo "::error::packaged crate is missing $label (pattern: $pattern)"
    exit 1
  fi
  echo "$label: $count"
}

require_any "build.rs" '(^|/)build\.rs$'
require_any "CUDA kernel sources (*.cu)" '\.cu$'
require_any "CUDA kernel headers (*.cuh)" '\.cuh$'
require_any "WGSL shaders (*.wgsl)" '\.wgsl$'
require_any "sobol_data.bin" 'src/runtime/cpu/kernels/sobol_data\.bin$'
require_any "CHANGELOG.md" '(^|/)CHANGELOG\.md$'
require_any "README.md" '(^|/)README\.md$'
require_any "LICENSE" '(^|/)LICENSE$'

echo "Package contents: ok"
