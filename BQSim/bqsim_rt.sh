#!/bin/bash
set -euo pipefail

# An empty CUDA_VISIBLE_DEVICES breaks CUDA initialization.
if [[ "${CUDA_VISIBLE_DEVICES-}" == "" ]]; then
  unset CUDA_VISIBLE_DEVICES
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build-rt"

## == gate fusion parts ==
: "${BQSIM_RT_PIPELINE_MODE:=SPMSPM}" # using RTSpMSpM to fuse gate other than dd

## == dense matrix calculation parts (SPMV/GEMV)==
: "${BQSIM_RT_HYBRID_DENSE:=1}"
: "${BQSIM_RT_DENSE_MAX_BYTES:=5368709120}" # 5120MB, the maximum size of dense matrix to use dense representation(for fear OOM)

## == gpu kernel execution parts ==
: "${BQSIM_RT_DENSE_TILE:=256}" # the tile size for dense matrix operations
: "${BQSIM_RT_DENSE_ASSUME_DENSE:=0}" # if set to 1, assume all matrices are dense(1 - use dense representation always)

## == kernel launch optimizations (do not need to modify this part)==
: "${BQSIM_RT_COMPACT_LAUNCH:=1}" # launch kernels directly
: "${BQSIM_RT_USE_CUDA_GRAPH:=1}" # Records all gate kernels into a graph to eliminate CPU-to-GPU launch overhead.
# (cuda graph reduce launch overhead, but increases GPU memory usage)
: "${BQSIM_RT_MEGA_KERNEL:=0}" # Runs all gates in ONE kernel using grid-wide sync (requires GPU capacity for all blocks).

## == GAS/BVH update strategy ==
: "${BQSIM_RT_GAS_ALLOW_UPDATE:=1}" # 1: allow OptiX GAS update when primitive count unchanged.
: "${BQSIM_RT_GAS_UPDATE_INTERVAL:=16}" # force rebuild after this many consecutive updates (0 disables).
: "${BQSIM_RT_GAS_UPDATE_MIN_PRIMS:=0}" # if prim count < this value, prefer rebuild over update.
: "${BQSIM_RT_GAS_REUSE_OUTPUT_BUFFER:=1}" # reuse GAS output buffer across rebuilds to reduce cudaMalloc/cudaFree.

export BQSIM_RT_PIPELINE_MODE
export BQSIM_RT_FUSED_GATE_SPM
export BQSIM_RT_DENSITY_TARGET
export BQSIM_RT_HYBRID_DENSE
export BQSIM_RT_DENSE_GEMV
export BQSIM_RT_DENSE_THRESHOLD
export BQSIM_RT_DENSE_MAX_BYTES
export BQSIM_RT_DENSE_TILE
export BQSIM_RT_DENSE_ASSUME_DENSE
export BQSIM_RT_COMPACT_LAUNCH
export BQSIM_RT_USE_CUDA_GRAPH
export BQSIM_RT_MEGA_KERNEL
export BQSIM_RT_GAS_ALLOW_UPDATE
export BQSIM_RT_GAS_UPDATE_INTERVAL
export BQSIM_RT_GAS_UPDATE_MIN_PRIMS
export BQSIM_RT_GAS_REUSE_OUTPUT_BUFFER

if [[ ! -x "${BUILD_DIR}/apps/BQSim" ]]; then
  echo "[bqsim_rt.sh] Missing ${BUILD_DIR}/apps/BQSim. Run: bash ${ROOT_DIR}/rt_compile.sh" >&2
  exit 1
fi

mkdir -p "${ROOT_DIR}/log/results/state"
mkdir -p "${ROOT_DIR}/log/fused_gates"

cd "${BUILD_DIR}/apps"


# the harder testcases
./BQSim --ps --pv --batch_size 256 --file ../../circuits/tsp_n9.qasm --num_batch 200 --conversion_type 2
./BQSim --ps --pv --batch_size 256 --file ../../circuits/tsp_n16.qasm --num_batch 200 --conversion_type 2
./BQSim --ps --pv --batch_size 256 --file ../../circuits/vqe_n12.qasm --num_batch 200 --conversion_type 2
./BQSim --ps --pv --batch_size 256 --file ../../circuits/vqe_n14.qasm --num_batch 200 --conversion_type 2
./BQSim --ps --pv --batch_size 256 --file ../../circuits/vqe_n16.qasm --num_batch 200 --conversion_type 2
./BQSim --ps --pv --batch_size 256 --file ../../circuits/routing_n6.qasm --num_batch 200 --conversion_type 2
./BQSim --ps --pv --batch_size 256 --file ../../circuits/routing_n12.qasm --num_batch 200 --conversion_type 2
./BQSim --ps --pv --batch_size 256 --file ../../circuits/portfolio_vqe_n16.qasm --num_batch 200 --conversion_type 2
./BQSim --ps --pv --batch_size 256 --file ../../circuits/portfolio_vqe_n17.qasm --num_batch 200 --conversion_type 2
./BQSim --ps --pv --batch_size 256 --file ../../circuits/portfolio_vqe_n18.qasm --num_batch 200 --conversion_type 2
./BQSim --ps --pv --batch_size 256 --file ../../circuits/graph_state_n16.qasm --num_batch 200 --conversion_type 2
./BQSim --ps --pv --batch_size 256 --file ../../circuits/graph_state_n18.qasm --num_batch 200 --conversion_type 2
# /BQSim --ps --pv --batch_size 256 --file ../../circuits/graph_state_n20.qasm --num_batch 200 --conversion_type 2
./BQSim --ps --pv --batch_size 256 --file ../../circuits/dnn_n17.qasm --num_batch 200 --conversion_type 2
./BQSim --ps --pv --batch_size 256 --file ../../circuits/dnn_n19.qasm --num_batch 200 --conversion_type 2
./BQSim --ps --pv --batch_size 256 --file ../../circuits/dnn_n21.qasm --num_batch 200 --conversion_type 2
