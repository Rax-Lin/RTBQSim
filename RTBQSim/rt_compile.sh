#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build-rt"

clear_build_dir() {
  if [[ -d "${BUILD_DIR}" ]]; then
    find "${BUILD_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  fi
}

resolve_cuda_arch() {
  if [[ -n "${CMAKE_CUDA_ARCHITECTURES:-}" ]]; then
    echo "[rt_compile.sh] Using user-provided CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}" >&2
    printf '%s\n' "${CMAKE_CUDA_ARCHITECTURES}"
    return 0
  fi

  local gpu_arch=""
  local gpu_cc=""
  local selected=""
  local arch=""
  local -a supported_arches=()

  if command -v nvcc >/dev/null 2>&1; then
    mapfile -t supported_arches < <(
      nvcc --list-gpu-arch 2>/dev/null \
        | sed -n 's/^compute_//p' \
        | grep -E '^[0-9]+$' \
        | sort -n -u
    )
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]' || true)"
    if [[ -n "${gpu_cc}" ]]; then
      gpu_arch="${gpu_cc/./}"
    fi
  fi

  if [[ -n "${gpu_arch}" && ${#supported_arches[@]} -gt 0 ]]; then
    for arch in "${supported_arches[@]}"; do
      if [[ "${arch}" == "${gpu_arch}" ]]; then
        selected="${arch}"
        break
      fi
      if (( 10#${arch} <= 10#${gpu_arch} )); then
        selected="${arch}"
      fi
    done

    if [[ -z "${selected}" ]]; then
      selected="${supported_arches[0]}"
      echo "[rt_compile.sh] GPU compute capability (${gpu_cc}) is below nvcc minimum; using ${selected}." >&2
    elif [[ "${selected}" != "${gpu_arch}" ]]; then
      echo "[rt_compile.sh] GPU compute capability (${gpu_cc}) not directly supported by nvcc; using compatible arch ${selected}." >&2
    else
      echo "[rt_compile.sh] Auto-detected CUDA arch ${selected} from GPU compute capability ${gpu_cc}." >&2
    fi

    printf '%s\n' "${selected}"
    return 0
  fi

  if [[ ${#supported_arches[@]} -gt 0 ]]; then
    selected="${supported_arches[${#supported_arches[@]}-1]}"
    echo "[rt_compile.sh] Could not query GPU compute capability; using highest nvcc-supported arch ${selected}." >&2
    printf '%s\n' "${selected}"
    return 0
  fi

  echo "[rt_compile.sh] Could not auto-detect CUDA arch; fallback to 86." >&2
  printf '%s\n' "86"
}

# Allow overrides via env
OPTIX_INSTALL_DIR="${OptiX_INSTALL_DIR:-/home/gpulabgogo/Optix/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64}"
CUDA_ARCH="$(resolve_cuda_arch)"
# OptiX desktop stacks can reject PTX .target sm_87 (Jetson-oriented arch).
# Prefer sm_86 fallback when auto-detection lands on 87.
if [[ "${CUDA_ARCH}" == "87" ]]; then
  echo "[rt_compile.sh] OptiX desktop compatibility fallback: sm_87 -> sm_86." >&2
  CUDA_ARCH="86"
fi
CUDA_HOST_COMPILER="${CMAKE_CUDA_HOST_COMPILER:-/usr/bin/gcc-9}"
CC_BIN="${CMAKE_C_COMPILER:-/usr/bin/gcc-9}"
CXX_BIN="${CMAKE_CXX_COMPILER:-/usr/bin/g++-9}"
CUQUANTUM_ROOT="${CUQUANTUM_ROOT:-/home/gpulabgogo/RTBQSim}"

if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  CACHE_SRC="$(grep -E '^CMAKE_HOME_DIRECTORY:' "${BUILD_DIR}/CMakeCache.txt" | cut -d= -f2- || true)"
  CACHE_ARCH="$(grep -E '^CMAKE_CUDA_ARCHITECTURES:' "${BUILD_DIR}/CMakeCache.txt" | cut -d= -f2- || true)"
  if [[ -n "${CACHE_SRC}" && "${CACHE_SRC}" != "${ROOT_DIR}" ]]; then
    echo "[rt_compile.sh] Detected mismatched CMake cache (${CACHE_SRC}). Cleaning ${BUILD_DIR}."
    clear_build_dir
  elif [[ -n "${CACHE_ARCH}" && "${CACHE_ARCH}" != "${CUDA_ARCH}" ]]; then
    echo "[rt_compile.sh] CUDA arch changed (${CACHE_ARCH} -> ${CUDA_ARCH}). Cleaning ${BUILD_DIR}."
    clear_build_dir
  fi
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
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

# Build only the RTBQSim target to avoid optional cuQuantum test build failures.
cmake --build "${BUILD_DIR}" --target RTBQSim -j
