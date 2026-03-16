#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${RTBQSIM_IMAGE:-rtbqsim-dev}"
BUILD_VOLUME="${RTBQSIM_BUILD_VOLUME:-rtbqsim-build}"
OPTIX_HOST_DIR=""
OPTIX_SEARCH_DEPTH="${RTBQSIM_OPTIX_SEARCH_DEPTH:-6}"

DO_BUILD=0
AUTO_RUN=0
declare -a USER_CMD=()

usage() {
  cat <<'EOF'
Usage: run_docker.sh [options] [-- command...]

Options:
  --build        Build docker image before running.
  --auto-run     Run: bash BQSim/rt_compile.sh && cd BQSim && bash bqsim_rt.sh
  -h, --help     Show this help.

Environment overrides:
  RTBQSIM_IMAGE         Docker image name (default: rtbqsim-dev)
  RTBQSIM_BUILD_VOLUME  Docker volume for build dir (default: rtbqsim-build)
  RTBQSIM_OPTIX_DIR     Host OptiX SDK root path (highest priority)
  RTBQSIM_OPTIX_SEARCH_DEPTH  Max recursive depth for auto-discovery (default: 6)
  BQSIM_RT_NUMERIC_PRECISION  Optional pass-through (fp32 or fp64) to container
EOF
}

is_optix_root_dir() {
  local dir="$1"
  [[ -n "${dir}" && -d "${dir}" && -f "${dir}/include/optix.h" ]]
}

is_optix_sdk_dir() {
  local dir="$1"
  [[ -n "${dir}" && -d "${dir}" && -f "${dir}/include/optix.h" && -f "${dir}/SDK/sutil/Preprocessor.h" ]]
}

normalize_optix_dir() {
  local input="$1"
  local candidate="${input%/}"

  if is_optix_sdk_dir "${candidate}"; then
    printf '%s\n' "${candidate}"
    return 0
  fi

  if [[ -d "${candidate}" && -f "${candidate}/optix.h" ]]; then
    candidate="$(dirname "${candidate}")"
    if is_optix_sdk_dir "${candidate}"; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  fi

  if [[ -f "${candidate}" && "$(basename "${candidate}")" == "optix.h" ]]; then
    candidate="$(dirname "$(dirname "${candidate}")")"
    if is_optix_sdk_dir "${candidate}"; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  fi

  return 1
}

discover_optix_dir() {
  local -a roots=()
  local -a candidates=()
  local -a sdk_candidates=()
  local current="${ROOT_DIR}"
  local parent=""
  local root=""
  local hit=""

  while true; do
    roots+=("${current}")
    [[ "${current}" == "/" ]] && break
    parent="$(dirname "${current}")"
    [[ "${parent}" == "${current}" ]] && break
    current="${parent}"
  done

  if [[ -n "${HOME:-}" ]]; then
    roots+=("${HOME}")
  fi
  if [[ -n "${USER:-}" ]]; then
    roots+=("/home/${USER}")
  fi
  roots+=("/opt" "/usr/local")

  while IFS= read -r root; do
    [[ -z "${root}" || ! -d "${root}" || "${root}" == "/" ]] && continue
    while IFS= read -r hit; do
      candidates+=("$(dirname "$(dirname "${hit}")")")
    done < <(find "${root}" -maxdepth "${OPTIX_SEARCH_DEPTH}" -type f -path '*/include/optix.h' 2>/dev/null)
  done < <(printf '%s\n' "${roots[@]}" | awk '!seen[$0]++')

  if [[ ${#candidates[@]} -eq 0 ]]; then
    return 1
  fi

  while IFS= read -r root; do
    if is_optix_sdk_dir "${root}"; then
      sdk_candidates+=("${root}")
    fi
  done < <(printf '%s\n' "${candidates[@]}" | awk '!seen[$0]++')

  if [[ ${#sdk_candidates[@]} -gt 0 ]]; then
    printf '%s\n' "${sdk_candidates[@]}" | awk '!seen[$0]++' | sort -V | tail -n 1
    return 0
  fi

  printf '%s\n' "${candidates[@]}" | awk '!seen[$0]++' | sort -V | tail -n 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)
      DO_BUILD=1
      shift
      ;;
    --auto-run)
      AUTO_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      USER_CMD=("$@")
      break
      ;;
    *)
      USER_CMD=("$@")
      break
      ;;
  esac
done

if [[ -n "${RTBQSIM_OPTIX_DIR:-}" ]]; then
  if ! OPTIX_HOST_DIR="$(normalize_optix_dir "${RTBQSIM_OPTIX_DIR}")"; then
    echo "[run_docker.sh] Invalid RTBQSIM_OPTIX_DIR: ${RTBQSIM_OPTIX_DIR}" >&2
    echo "[run_docker.sh] Expected full OptiX SDK root containing include/optix.h and SDK/sutil/Preprocessor.h." >&2
    exit 1
  fi
else
  if ! OPTIX_HOST_DIR="$(discover_optix_dir)"; then
    echo "[run_docker.sh] OptiX SDK not found via auto-discovery." >&2
    echo "[run_docker.sh] Searched recursively from parent dirs of: ${ROOT_DIR}" >&2
    echo "[run_docker.sh] Set RTBQSIM_OPTIX_DIR to your local OptiX SDK root path." >&2
    echo "[run_docker.sh] Expected target files: <OptiX_ROOT>/include/optix.h and <OptiX_ROOT>/SDK/sutil/Preprocessor.h" >&2
    exit 1
  fi
  echo "[run_docker.sh] Auto-detected OptiX SDK: ${OPTIX_HOST_DIR}"
fi

if ! is_optix_sdk_dir "${OPTIX_HOST_DIR}"; then
  echo "[run_docker.sh] OptiX dir not valid: ${OPTIX_HOST_DIR}" >&2
  echo "[run_docker.sh] Expected target files: <OptiX_ROOT>/include/optix.h and <OptiX_ROOT>/SDK/sutil/Preprocessor.h" >&2
  exit 1
fi

if [[ "${DO_BUILD}" -eq 1 ]]; then
  docker build -t "${IMAGE_NAME}" -f "${ROOT_DIR}/Dockerfile" "${ROOT_DIR}"
fi

# Initialize/chown build volume so non-root container user can write build outputs.
docker run --rm -v "${BUILD_VOLUME}:/build" alpine:3.20 \
  sh -lc "mkdir -p /build && chown -R $(id -u):$(id -g) /build" >/dev/null

declare -a RUN_ARGS=(
  --rm
  --gpus all
  -e NVIDIA_DRIVER_CAPABILITIES=all
  -e HOME=/tmp
  --user "$(id -u):$(id -g)"
  -v "${ROOT_DIR}:/workspace/RT_BQSim"
  -v "${BUILD_VOLUME}:/workspace/RT_BQSim/BQSim/build-rt"
  -v "${OPTIX_HOST_DIR}:/opt/optix:ro"
  -w /workspace/RT_BQSim
)

if [[ -n "${BQSIM_RT_NUMERIC_PRECISION:-}" ]]; then
  if [[ "${BQSIM_RT_NUMERIC_PRECISION}" != "fp32" && "${BQSIM_RT_NUMERIC_PRECISION}" != "fp64" ]]; then
    echo "[run_docker.sh] BQSIM_RT_NUMERIC_PRECISION must be fp32 or fp64 (got: ${BQSIM_RT_NUMERIC_PRECISION})" >&2
    exit 1
  fi
  RUN_ARGS+=(-e "BQSIM_RT_NUMERIC_PRECISION=${BQSIM_RT_NUMERIC_PRECISION}")
fi

for host_lib_dir in /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu; do
  if [[ -f "${host_lib_dir}/libnvoptix.so" ]]; then
    RUN_ARGS+=(-v "${host_lib_dir}/libnvoptix.so:/usr/lib/libnvoptix.so:ro")
    break
  fi
done

for host_lib_dir in /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu; do
  if [[ -f "${host_lib_dir}/libnvoptix.so.1" ]]; then
    RUN_ARGS+=(-v "${host_lib_dir}/libnvoptix.so.1:/usr/lib/libnvoptix.so.1:ro")
    # Some systems only expose .so.1; also map it as .so for find_library(name=nvoptix).
    if ! [[ " ${RUN_ARGS[*]} " =~ /usr/lib/libnvoptix\.so:ro ]]; then
      RUN_ARGS+=(-v "${host_lib_dir}/libnvoptix.so.1:/usr/lib/libnvoptix.so:ro")
    fi
    break
  fi
done

if [[ "${AUTO_RUN}" -eq 1 ]]; then
  USER_CMD=(bash -lc "bash BQSim/rt_compile.sh && cd BQSim && bash bqsim_rt.sh")
fi

if [[ ${#USER_CMD[@]} -eq 0 ]]; then
  docker run -it "${RUN_ARGS[@]}" "${IMAGE_NAME}" bash
else
  docker run "${RUN_ARGS[@]}" "${IMAGE_NAME}" "${USER_CMD[@]}"
fi
