#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

clear_build_dir() {
  if [[ -d "${BUILD_DIR}" ]]; then
    find "${BUILD_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  fi
}

# Toolchain/env defaults (override with env vars when needed)
CUDA_HOST_COMPILER="${CMAKE_CUDA_HOST_COMPILER:-/usr/bin/gcc-9}"
CC_BIN="${CMAKE_C_COMPILER:-/usr/bin/gcc-9}"
CXX_BIN="${CMAKE_CXX_COMPILER:-/usr/bin/g++-9}"
CUQUANTUM_ROOT="${CUQUANTUM_ROOT:-/usr/local/cuquantum}"

# Keep compatibility with existing CMake logic that reads env CUSTATEVEC_LIBRARY.
: "${CUSTATEVEC_LIBRARY:=${CUQUANTUM_ROOT}/lib/libcustatevec.so}"
export CUSTATEVEC_LIBRARY
export CUQUANTUM_ROOT

if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  CACHE_SRC="$(grep -E '^CMAKE_HOME_DIRECTORY:' "${BUILD_DIR}/CMakeCache.txt" | cut -d= -f2- || true)"
  if [[ -n "${CACHE_SRC}" && "${CACHE_SRC}" != "${ROOT_DIR}" ]]; then
    echo "[cuquantum_compile.sh] Detected mismatched CMake cache (${CACHE_SRC}). Cleaning ${BUILD_DIR}."
    clear_build_dir
  fi
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_HOST_COMPILER="${CUDA_HOST_COMPILER}" \
  -DCMAKE_C_COMPILER="${CC_BIN}" \
  -DCMAKE_CXX_COMPILER="${CXX_BIN}" \
  -DCUQUANTUM_ROOT="${CUQUANTUM_ROOT}"

# Build only cuquantum target used by cuquantum.sh
cmake --build "${BUILD_DIR}" --target cuquantum -j

echo "[cuquantum_compile.sh] Built ${BUILD_DIR}/cuquantum_test/cuquantum"
