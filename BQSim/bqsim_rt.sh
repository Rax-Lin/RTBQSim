#!/bin/bash
set -euo pipefail

# An empty CUDA_VISIBLE_DEVICES breaks CUDA initialization.
if [[ "${CUDA_VISIBLE_DEVICES-}" == "" ]]; then
  unset CUDA_VISIBLE_DEVICES
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${BQSIM_RT_NUMERIC_PRECISION:=fp64}" # fp32 or fp64 (applies to stage-1 + stage-2 numeric type)
if [[ "${BQSIM_RT_NUMERIC_PRECISION}" != "fp32" && "${BQSIM_RT_NUMERIC_PRECISION}" != "fp64" ]]; then
  echo "[bqsim_rt.sh] BQSIM_RT_NUMERIC_PRECISION must be fp32 or fp64 (got: ${BQSIM_RT_NUMERIC_PRECISION})" >&2
  exit 1
fi
BUILD_DIR="${ROOT_DIR}/build-rt"

## == gate fusion parts ==
: "${BQSIM_RT_PIPELINE_MODE:=SPMSPM}" # using RTSpMSpM to fuse gate other than dd

## == fusion stop behavior ==
: "${BQSIM_RT_FORCE_FULL_FUSION:=0}" # 1: do not early-stop by row nnz limit; fuse until max gates.

## == GAS/BVH update strategy ==
: "${BQSIM_RT_GAS_ALLOW_UPDATE:=0}" # 1: allow OptiX GAS update when primitive count unchanged.
: "${BQSIM_RT_GAS_UPDATE_INTERVAL:=0}" # force rebuild after this many consecutive updates (0 disables). This is to prevent too many updates when the circuit has many similar segments.
: "${BQSIM_RT_GAS_REUSE_OUTPUT_BUFFER:=1}" # reuse GAS output buffer across rebuilds to reduce cudaMalloc/cudaFree.
: "${BQSIM_RT_REUSE_GEOMETRY_BUFFER:=${BQSIM_RT_GAS_REUSE_OUTPUT_BUFFER}}" # reuse sphere/ray-side geometry work buffers; defaults to GAS output reuse setting.

## == diagonal gate optimization ==
: "${BQSIM_RT_DIAG_VALUE_ONLY:=1}" # 1: diagonal gates only update values (keep row/col topology).

## == stage-1 refit shift metric ==
: "${BQSIM_RT_REFIT_SHIFT_METRIC:=0}" # 1: enable primitive-position shift metric against latest rebuild baseline.

## == stage-1 gate dump ==
: "${BQSIM_RT_DUMP_BUILD_GATES:=1}" # 1: dump only build/rebuild-associated primitive gates to log/build_gate/<circuit>_primitive_gates.csv.

## == stage-1 pure timing mode ==
: "${BQSIM_RT_SYNC_STAGE_TIMING:=1}" # 1: use CUDA event+synchronize to measure pure stage times; breakdown sum may exceed Stage-1 wall time due to overlap.

## == stage-1 stream scheduling ==
: "${BQSIM_RT_SERIAL_PREP_STREAM:=1}" # 1: reduce prep_stream/main-stream overlap by synchronizing before gate preparation; improves Ray Generation measurement fidelity, but does not make Stage-1 fully serial.

export BQSIM_RT_PIPELINE_MODE
export BQSIM_RT_FORCE_FULL_FUSION
export BQSIM_RT_GAS_ALLOW_UPDATE
export BQSIM_RT_GAS_UPDATE_INTERVAL
export BQSIM_RT_GAS_REUSE_OUTPUT_BUFFER
export BQSIM_RT_REUSE_GEOMETRY_BUFFER
export BQSIM_RT_DIAG_VALUE_ONLY
export BQSIM_RT_REFIT_SHIFT_METRIC
export BQSIM_RT_DUMP_BUILD_GATES
export BQSIM_RT_SYNC_STAGE_TIMING
export BQSIM_RT_SERIAL_PREP_STREAM
export BQSIM_RT_NUMERIC_PRECISION

echo "[bqsim_rt.sh] Numeric precision: ${BQSIM_RT_NUMERIC_PRECISION}"

needs_compile=0
if [[ ! -x "${BUILD_DIR}/apps/BQSim" ]]; then
  needs_compile=1
  echo "[bqsim_rt.sh] Missing ${BUILD_DIR}/apps/BQSim, building (${BQSIM_RT_NUMERIC_PRECISION})..."
elif [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  cache_precision="$(grep -E '^BQSIM_RT_NUMERIC_PRECISION:' "${BUILD_DIR}/CMakeCache.txt" | cut -d= -f2- || true)"
  cache_arch="$(grep -E '^CMAKE_CUDA_ARCHITECTURES:' "${BUILD_DIR}/CMakeCache.txt" | cut -d= -f2- || true)"
  if [[ -z "${cache_precision}" || "${cache_precision}" != "${BQSIM_RT_NUMERIC_PRECISION}" ]]; then
    needs_compile=1
    echo "[bqsim_rt.sh] build-rt precision mismatch (${cache_precision:-unknown} -> ${BQSIM_RT_NUMERIC_PRECISION}), rebuilding..."
  elif [[ "${cache_arch}" == "87" ]]; then
    needs_compile=1
    echo "[bqsim_rt.sh] build-rt uses CUDA arch sm_87 (OptiX incompatible here), rebuilding with fallback arch..."
  fi
else
  needs_compile=1
  echo "[bqsim_rt.sh] Missing ${BUILD_DIR}/CMakeCache.txt, rebuilding..."
fi

if [[ "${needs_compile}" -eq 1 ]]; then
  bash "${ROOT_DIR}/rt_compile.sh"
fi

verify() { python3 "${ROOT_DIR}/verify.py" -c "$1" -n "$2"; }

mkdir -p "${ROOT_DIR}/log/results/state"
mkdir -p "${ROOT_DIR}/log/fused_gates"
mkdir -p "${ROOT_DIR}/log/build_gate"

cd "${BUILD_DIR}/apps"


# the harder testcases
./BQSim --ps --pv --batch_size 32 --file ../../circuits/random_n19.qasm --num_batch 10 --conversion_type 2
./BQSim --ps --pv --batch_size 32 --file ../../circuits/random_n20.qasm --num_batch 10 --conversion_type 2
#./BQSim --ps --pv --batch_size 32 --file ../../circuits/random_n21.qasm --num_batch 10 --conversion_type 2
# ./BQSim --ps --pv --batch_size 32 --file ../../circuits/qnn_n23.qasm --num_batch 10 --conversion_type 2
# ./BQSim --ps --pv --batch_size 32 --file ../../circuits/tsp_n9.qasm --num_batch 10 --conversion_type 2
./BQSim --ps --pv --batch_size 32 --file ../../circuits/tsp_n16.qasm --num_batch 10 --conversion_type 2
# ./BQSim --ps --pv --batch_size 32 --file ../../circuits/vqe_n12.qasm --num_batch 10 --conversion_type 2
# ./BQSim --ps --pv --batch_size 32 --file ../../circuits/vqe_n14.qasm --num_batch 10 --conversion_type 2
./BQSim --ps --pv --batch_size 32 --file ../../circuits/vqe_n16.qasm --num_batch 10 --conversion_type 2
# ./BQSim --ps --pv --batch_size 32 --file ../../circuits/routing_n6.qasm --num_batch 10 --conversion_type 2
# ./BQSim --ps --pv --batch_size 32 --file ../../circuits/routing_n12.qasm --num_batch 10 --conversion_type 2
./BQSim --ps --pv --batch_size 32 --file ../../circuits/portfolio_vqe_n16.qasm --num_batch 10 --conversion_type 2
./BQSim --ps --pv --batch_size 32 --file ../../circuits/portfolio_vqe_n17.qasm --num_batch 10 --conversion_type 2
./BQSim --ps --pv --batch_size 32 --file ../../circuits/portfolio_vqe_n18.qasm --num_batch 10 --conversion_type 2
./BQSim --ps --pv --batch_size 32 --file ../../circuits/graph_state_n16.qasm --num_batch 10 --conversion_type 2
./BQSim --ps --pv --batch_size 32 --file ../../circuits/graph_state_n18.qasm --num_batch 10 --conversion_type 2
./BQSim --ps --pv --batch_size 32 --file ../../circuits/graph_state_n20.qasm --num_batch 10 --conversion_type 2
./BQSim --ps --pv --batch_size 32 --file ../../circuits/graph_state_n22.qasm --num_batch 10 --conversion_type 2

./BQSim --ps --pv --batch_size 32 --file ../../circuits/dnn_n17.qasm --num_batch 10 --conversion_type 2
./BQSim --ps --pv --batch_size 32 --file ../../circuits/dnn_n19.qasm --num_batch 10 --conversion_type 2
./BQSim --ps --pv --batch_size 32 --file ../../circuits/dnn_n21.qasm --num_batch 10 --conversion_type 2

verify random 19
#verify random 20
#verify random 21
# verify qnn 23
verify tsp 9
verify tsp 16
verify vqe 12
verify vqe 14
verify vqe 16
verify routing 6
verify routing 12
verify portfolio_vqe 16
verify portfolio_vqe 17
verify portfolio_vqe 18
verify graph_state 16
verify graph_state 18
verify graph_state 20
verify graph_state 22
verify graph_state 23
verify dnn 17
verify dnn 19
verify dnn 21
