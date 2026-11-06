#!/usr/bin/env bash
# Build and run a kernel.cu headlessly on the CPU (no GPU, no window).
#   usage: test/run_headless.sh [path/to/kernel.cu] [out.csv]
# The SHIM_* environment variables in shim/raylib.h control frames, mouse
# waypoints and key presses. Extra compiler flags can go in CXXFLAGS.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
src="${1:-$here/../CudaRuntime5/kernel.cu}"
out="${2:-}"
build="$(mktemp -d)"
trap 'rm -rf "$build"' EXIT
# Turn CUDA launch syntax  k<<<grid, block>>>(args)  (with any spacing)  into  SHIM_LAUNCH(k, grid, block)(args)
perl -0pe 's/(\w+)\s*<<\s*<\s*([^,>]+?)\s*,\s*([^>]+?)\s*>>\s*>\s*\(/SHIM_LAUNCH($1, $2, $3)(/g' "$src" > "$build/kernel.cpp"
g++ -std=c++17 -O2 ${CXXFLAGS:-} -I"$here/shim" "$build/kernel.cpp" -o "$build/kernel"
if [[ -n "$out" ]]; then SHIM_CSV="$out" "$build/kernel"; else "$build/kernel"; fi
