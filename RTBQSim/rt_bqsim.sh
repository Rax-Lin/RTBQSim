#!/bin/bash
set -euo pipefail

# An empty CUDA_VISIBLE_DEVICES breaks CUDA initialization.
if [[ "${CUDA_VISIBLE_DEVICES-}" == "" ]]; then
  unset CUDA_VISIBLE_DEVICES
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build-rt"

## == GAS/BVH update strategy ==
: "${RT_GAS_ALLOW_UPDATE:=1}" # 1: allow OptiX GAS update when primitive count unchanged.
: "${RT_REUSE_BUFFER:=1}" # 1: reuse GAS output + sphere/ray geometry work buffers to reduce cudaMalloc/cudaFree.
: "${RT_DIAG_VALUE_ONLY:=1}" # 1: diagonal gates only update values (keep row/col topology). Disabled by default for correctness.

## == stage-1 traversal CSV dumps ==
: "${RT_DUMP_TREE_OWNER_AVG:=0}" # 1: dump build/rebuild-associated tree-owner gates with traversal averages to log/{refit,no_refit}_tree_owner/<circuit>_primitive_gates.csv.
: "${RT_DUMP_GATE_TRAVERSAL:=0}" # 1: dump every fused-block gate's pure traversal time to log/{refit,no_refit}_per_gate/<circuit>_per_gate.csv.

export RT_GAS_ALLOW_UPDATE
export RT_REUSE_BUFFER
export RT_DIAG_VALUE_ONLY
export RT_DUMP_TREE_OWNER_AVG
export RT_DUMP_GATE_TRAVERSAL

echo "[rt_bqsim.sh] Numeric precision: fp64 (fixed)"

needs_compile=0
if [[ ! -x "${BUILD_DIR}/apps/RTBQSim" ]]; then
  needs_compile=1
  echo "[rt_bqsim.sh] Missing ${BUILD_DIR}/apps/RTBQSim, building..."
elif [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  cache_arch="$(grep -E '^CMAKE_CUDA_ARCHITECTURES:' "${BUILD_DIR}/CMakeCache.txt" | cut -d= -f2- || true)"
  if [[ "${cache_arch}" == "87" ]]; then
    needs_compile=1
    echo "[rt_bqsim.sh] build-rt uses CUDA arch sm_87 (OptiX incompatible here), rebuilding with fallback arch..."
  fi
else
  needs_compile=1
  echo "[rt_bqsim.sh] Missing ${BUILD_DIR}/CMakeCache.txt, rebuilding..."
fi

if [[ "${needs_compile}" -eq 1 ]]; then
  bash "${ROOT_DIR}/rt_compile.sh"
fi

verify() { python3 "${ROOT_DIR}/verify.py" -c "$1" -n "$2"; }

mkdir -p "${ROOT_DIR}/log/results/state"
mkdir -p "${ROOT_DIR}/log/refit_tree_owner"
mkdir -p "${ROOT_DIR}/log/no_refit_tree_owner"
mkdir -p "${ROOT_DIR}/log/refit_per_gate"
mkdir -p "${ROOT_DIR}/log/no_refit_per_gate"

cd "${BUILD_DIR}/apps"


# the harder testcases
./RTBQSim --ps --pv --batch_size 32 --file ../../circuits/random_n19.qasm --num_batch 10 --conversion_type 2
./RTBQSim --ps --pv --batch_size 32 --file ../../circuits/random_n20.qasm --num_batch 10 --conversion_type 2
#./RTBQSim --ps --pv --batch_size 32 --file ../../circuits/random_n21.qasm --num_batch 10 --conversion_type 2
# ./RTBQSim --ps --pv --batch_size 32 --file ../../circuits/qnn_n23.qasm --num_batch 10 --conversion_type 2
#./RTBQSim --ps --pv --batch_size 32 --file ../../circuits/tsp_n9.qasm --num_batch 10 --conversion_type 2
./RTBQSim --ps --pv --batch_size 32 --file ../../circuits/tsp_n16.qasm --num_batch 10 --conversion_type 2
#./RTBQSim --ps --pv --batch_size 32 --file ../../circuits/vqe_n12.qasm --num_batch 10 --conversion_type 2
#./RTBQSim --ps --pv --batch_size 32 --file ../../circuits/vqe_n14.qasm --num_batch 10 --conversion_type 2
./RTBQSim --ps --pv --batch_size 32 --file ../../circuits/vqe_n16.qasm --num_batch 10 --conversion_type 2
#./RTBQSim --ps --pv --batch_size 32 --file ../../circuits/routing_n6.qasm --num_batch 10 --conversion_type 2
#./RTBQSim --ps --pv --batch_size 32 --file ../../circuits/routing_n12.qasm --num_batch 10 --conversion_type 2
./RTBQSim --ps --pv --batch_size 32 --file ../../circuits/portfolio_vqe_n16.qasm --num_batch 10 --conversion_type 2
./RTBQSim --ps --pv --batch_size 32 --file ../../circuits/portfolio_vqe_n17.qasm --num_batch 10 --conversion_type 2
./RTBQSim --ps --pv --batch_size 32 --file ../../circuits/portfolio_vqe_n18.qasm --num_batch 10 --conversion_type 2
./RTBQSim --ps --pv --batch_size 32 --file ../../circuits/graph_state_n16.qasm --num_batch 10 --conversion_type 2
./RTBQSim --ps --pv --batch_size 32 --file ../../circuits/graph_state_n18.qasm --num_batch 10 --conversion_type 2
./RTBQSim --ps --pv --batch_size 32 --file ../../circuits/graph_state_n20.qasm --num_batch 10 --conversion_type 2
#./RTBQSim --ps --pv --batch_size 32 --file ../../circuits/graph_state_n22.qasm --num_batch 10 --conversion_type 2
./RTBQSim --ps --pv --batch_size 32 --file ../../circuits/dnn_n17.qasm --num_batch 10 --conversion_type 2
./RTBQSim --ps --pv --batch_size 32 --file ../../circuits/dnn_n19.qasm --num_batch 10 --conversion_type 2
./RTBQSim --ps --pv --batch_size 32 --file ../../circuits/dnn_n21.qasm --num_batch 10 --conversion_type 2

verify random 19
verify random 20
#verify random 21
# verify qnn 23
# verify tsp 9
verify tsp 16
# verify vqe 12
verify vqe 14
# verify routing 6
# verify routing 12
verify portfolio_vqe 16
verify portfolio_vqe 17
verify portfolio_vqe 18
verify graph_state 16
verify graph_state 18
verify graph_state 20
#verify graph_state 22
#verify graph_state 23
verify dnn 17
verify dnn 19
verify dnn 21
