#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build-rt"

# Allow overrides via env
OPTIX_INSTALL_DIR="${OptiX_INSTALL_DIR:-/home/gpulabgogo/Optix/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64}"
CUDA_ARCH="${CMAKE_CUDA_ARCHITECTURES:-86}"
CUDA_HOST_COMPILER="${CMAKE_CUDA_HOST_COMPILER:-/usr/bin/gcc-9}"
CC_BIN="${CMAKE_C_COMPILER:-/usr/bin/gcc-9}"
CXX_BIN="${CMAKE_CXX_COMPILER:-/usr/bin/g++-9}"
CUQUANTUM_ROOT="${CUQUANTUM_ROOT:-/home/gpulabgogo/BQSim}"

if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  CACHE_SRC="$(grep -E '^CMAKE_HOME_DIRECTORY:' "${BUILD_DIR}/CMakeCache.txt" | cut -d= -f2- || true)"
  if [[ -n "${CACHE_SRC}" && "${CACHE_SRC}" != "${ROOT_DIR}" ]]; then
    echo "[rt_compile.sh] Detected mismatched CMake cache (${CACHE_SRC}). Cleaning ${BUILD_DIR}."
    rm -rf "${BUILD_DIR}"
  fi
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DBQSIM_USE_RTSPMSPM=ON \
  -DOptiX_INSTALL_DIR="${OPTIX_INSTALL_DIR}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
  -DCMAKE_CUDA_HOST_COMPILER="${CUDA_HOST_COMPILER}" \
  -DCMAKE_C_COMPILER="${CC_BIN}" \
  -DCMAKE_CXX_COMPILER="${CXX_BIN}" \
  -DOpenMP_CXX_FLAGS=-fopenmp \
  -DOpenMP_CXX_LIB_NAMES=gomp \
  -DOpenMP_gomp_LIBRARY=/usr/lib/x86_64-linux-gnu/libgomp.so.1 \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCUQUANTUM_ROOT="${CUQUANTUM_ROOT}"

# Build only the BQSim target to avoid optional cuQuantum test build failures.
cmake --build "${BUILD_DIR}" --target BQSim -j
