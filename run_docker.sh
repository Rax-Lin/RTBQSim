#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${RTBQSIM_IMAGE:-rtbqsim-dev}"
BUILD_VOLUME="${RTBQSIM_BUILD_VOLUME:-rtbqsim-build}"
OPTIX_HOST_DIR="${RTBQSIM_OPTIX_DIR:-/home/gpulabgogo/Optix/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64}"

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
  RTBQSIM_OPTIX_DIR     Host OptiX SDK path (default: /home/gpulabgogo/Optix/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64)
  BQSIM_RT_NUMERIC_PRECISION  Optional pass-through (fp32 or fp64) to container
EOF
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

if [[ ! -d "${OPTIX_HOST_DIR}" ]]; then
  echo "[run_docker.sh] OptiX dir not found: ${OPTIX_HOST_DIR}" >&2
  echo "[run_docker.sh] Set RTBQSIM_OPTIX_DIR to your local OptiX SDK path." >&2
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

if [[ -f /usr/lib/x86_64-linux-gnu/libnvoptix.so ]]; then
  RUN_ARGS+=(-v /usr/lib/x86_64-linux-gnu/libnvoptix.so:/usr/lib/libnvoptix.so:ro)
fi
if [[ -f /usr/lib/x86_64-linux-gnu/libnvoptix.so.1 ]]; then
  RUN_ARGS+=(-v /usr/lib/x86_64-linux-gnu/libnvoptix.so.1:/usr/lib/libnvoptix.so.1:ro)
fi

if [[ "${AUTO_RUN}" -eq 1 ]]; then
  USER_CMD=(bash -lc "bash BQSim/rt_compile.sh && cd BQSim && bash bqsim_rt.sh")
fi

if [[ ${#USER_CMD[@]} -eq 0 ]]; then
  docker run -it "${RUN_ARGS[@]}" "${IMAGE_NAME}" bash
else
  docker run "${RUN_ARGS[@]}" "${IMAGE_NAME}" "${USER_CMD[@]}"
fi
