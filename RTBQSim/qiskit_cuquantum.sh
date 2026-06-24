#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QISKIT_DIR="${ROOT_DIR}/qiskit_test"
CUQ_DIR="${ROOT_DIR}/build/cuquantum_test"
LOG_DIR="${ROOT_DIR}/log"
VENV_DIR="${ROOT_DIR}/.venv"
CUQUANTUM_VENV_LIB="${VENV_DIR}/lib/python3.12/site-packages/cuquantum/lib"
CUTENSOR_VENV_LIB="${VENV_DIR}/lib/python3.12/site-packages/cutensor/lib"

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

append_ld_library_path() {
  local dir="$1"
  if [[ -d "${dir}" ]]; then
    if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
      export LD_LIBRARY_PATH="${dir}:${LD_LIBRARY_PATH}"
    else
      export LD_LIBRARY_PATH="${dir}"
    fi
  fi
}

: "${QISKIT_FUSION_ENGINE:=transpiler}"
: "${QISKIT_FUSION_DEVICE:=gpu}"
: "${QISKIT_FUSION_THRESHOLD:=5}"
: "${QISKIT_FUSION_MAX_QUBIT:=5}"
: "${QISKIT_CPU_THREADS:=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 0)}"
: "${CUQ_BATCH_SIZE:=32}"
: "${CUQ_NUM_BATCH:=50}"
: "${CUQ_OUTPUT_STATE:=0}"
: "${CUQ_BINARY:=}"
: "${QISKIT_PYTHON_BIN:=}"

append_ld_library_path "${CUQUANTUM_VENV_LIB}"
append_ld_library_path "${CUTENSOR_VENV_LIB}"

mkdir -p "${LOG_DIR}"
mkdir -p "${ROOT_DIR}/log/fused_gates"
mkdir -p "${ROOT_DIR}/log/results/state"
mkdir -p "${CUQ_DIR}"

check_python_modules() {
  local python_bin="$1"
  shift
  "${python_bin}" - "$@" <<'PY'
import importlib
import sys

mods = tuple(sys.argv[1:])
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    raise SystemExit("Missing Python modules: " + ", ".join(missing))
PY
}

choose_python_bin() {
  local -a required_mods=("qiskit" "numpy")
  if [[ "${QISKIT_FUSION_ENGINE}" == "aer" ]]; then
    required_mods+=("qiskit_aer")
  fi

  local -a candidates=()
  if [[ -n "${QISKIT_PYTHON_BIN}" ]]; then
    candidates+=("${QISKIT_PYTHON_BIN}")
  fi
  candidates+=("${VENV_DIR}/bin/python" "python3" "python")

  local candidate
  local seen="|"
  for candidate in "${candidates[@]}"; do
    [[ -z "${candidate}" ]] && continue
    if [[ "${seen}" == *"|${candidate}|"* ]]; then
      continue
    fi
    seen="${seen}${candidate}|"
    if ! command -v "${candidate}" >/dev/null 2>&1; then
      continue
    fi
    if check_python_modules "${candidate}" "${required_mods[@]}" >/dev/null 2>&1; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

PYTHON_BIN="$(choose_python_bin || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
  required_modules_msg="qiskit, numpy"
  if [[ "${QISKIT_FUSION_ENGINE}" == "aer" ]]; then
    required_modules_msg+=", qiskit_aer"
  fi
  echo "[qiskit_cuquantum.sh] No usable Python interpreter with the required modules was found." >&2
  echo "[qiskit_cuquantum.sh] Required modules: ${required_modules_msg}" >&2
  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    echo "[qiskit_cuquantum.sh] Install them into the local venv with:" >&2
    echo "  ${VENV_DIR}/bin/python -m pip install --upgrade pip" >&2
    echo "  ${VENV_DIR}/bin/python -m pip install qiskit numpy qiskit-aer" >&2
  else
    echo "[qiskit_cuquantum.sh] Create a local venv and install them with:" >&2
    echo "  python3 -m venv ${VENV_DIR}" >&2
    echo "  ${VENV_DIR}/bin/python -m pip install --upgrade pip" >&2
    echo "  ${VENV_DIR}/bin/python -m pip install qiskit numpy qiskit-aer" >&2
  fi
  echo "[qiskit_cuquantum.sh] You can also override the interpreter with QISKIT_PYTHON_BIN=/path/to/python." >&2
  exit 1
fi

resolve_cuq_binary() {
  if [[ -n "${CUQ_BINARY}" && -x "${CUQ_BINARY}" ]]; then
    printf '%s\n' "${CUQ_BINARY}"
    return 0
  fi
  if [[ -x "${CUQ_DIR}/cuquantum" ]]; then
    printf '%s\n' "${CUQ_DIR}/cuquantum"
    return 0
  fi
  local fallback="/home/rtbqsim/0621/RTBQSim/RTBQSim/build/cuquantum_test/cuquantum"
  if [[ -x "${fallback}" ]]; then
    printf '%s\n' "${fallback}"
    return 0
  fi
  return 1
}

CUQ_RUNNER="$(resolve_cuq_binary || true)"
if [[ -z "${CUQ_RUNNER}" ]]; then
  echo "[qiskit_cuquantum.sh] No usable cuquantum binary found." >&2
  echo "[qiskit_cuquantum.sh] Set CUQ_BINARY=/path/to/cuquantum or build it in a CUDA-enabled environment." >&2
  exit 1
fi

run_export_case() {
  local mode="$1"
  local circuit="$2"
  local qubits="$3"
  "${PYTHON_BIN}" "${QISKIT_DIR}/qiskit_export_gate.py" \
    --circuit_name "${circuit}" \
    --num_qubits "${qubits}" \
    --fusion-engine "${mode}" \
    --device "${QISKIT_FUSION_DEVICE}" \
    --fusion-threshold "${QISKIT_FUSION_THRESHOLD}" \
    --fusion-max-qubit "${QISKIT_FUSION_MAX_QUBIT}" \
    --max-parallel-threads "${QISKIT_CPU_THREADS}" \
    --output-basename "qiskit_${circuit}_n${qubits}.txt"
}

run_cuquantum_gatefile_case() {
  local circuit="$1"
  local qubits="$2"
  (
    cd "${CUQ_DIR}"
    "${CUQ_RUNNER}" "${circuit}" "${qubits}" "${CUQ_BATCH_SIZE}" "${CUQ_NUM_BATCH}" 1 "${CUQ_OUTPUT_STATE}"
  )
}

extract_ms() {
  local label="$1"
  local text="$2"
  printf '%s\n' "${text}" | awk -v prefix="${label}: " '
    index($0, prefix) == 1 {
      print $(NF-1)
    }
  ' | tail -n1
}

extract_value() {
  local label="$1"
  local text="$2"
  printf '%s\n' "${text}" | awk -v prefix="${label}: " '
    index($0, prefix) == 1 {
      sub(prefix, "", $0)
      print $0
    }
  ' | tail -n1
}

print_case() {
  local circuit="$1"
  local qubits="$2"
  local fused_export_output
  local fused_export_gate_fusion_ms
  local fused_export_total_ms
  local fused_gatefile
  local fused_output
  local fused_core_ms
  local fused_simulation_ms
  local fused_bridge_ms
  local fused_total_ms

  fused_export_output="$(run_export_case "${QISKIT_FUSION_ENGINE}" "${circuit}" "${qubits}")"
  fused_export_gate_fusion_ms="$(extract_ms "Qiskit gate fusion time" "${fused_export_output}")"
  fused_export_total_ms="$(extract_ms "Qiskit pure fusion time" "${fused_export_output}")"
  fused_gatefile="$(extract_value "Fused gate file" "${fused_export_output}")"
  if [[ -z "${fused_export_gate_fusion_ms}" ]]; then
    fused_export_gate_fusion_ms="${fused_export_total_ms}"
  fi
  if [[ -z "${fused_export_gate_fusion_ms}" || -z "${fused_gatefile}" ]]; then
    echo "[qiskit_cuquantum.sh] Failed to export fused gate file for ${circuit}_n${qubits}." >&2
    return 1
  fi

  fused_output="$(run_cuquantum_gatefile_case "${circuit}" "${qubits}")"
  fused_core_ms="$(extract_ms "cuQuantum runtime" "${fused_output}")"
  fused_simulation_ms="$(extract_ms "cuQuantum simulation time" "${fused_output}")"
  fused_bridge_ms="$(extract_ms "cuQuantum fused bridge time" "${fused_output}")"
  if [[ -z "${fused_core_ms}" ]]; then
    echo "[qiskit_cuquantum.sh] Failed to parse fused-gate cuQuantum runtime for ${circuit}_n${qubits}." >&2
    return 1
  fi
  if [[ -z "${fused_simulation_ms}" ]]; then
    fused_simulation_ms="${fused_core_ms}"
  fi
  : "${fused_bridge_ms:=0.00}"

  fused_total_ms="$(python3 - <<PY
export_ms = float(${fused_export_gate_fusion_ms})
fused_ms = float(${fused_simulation_ms})
print(f"{export_ms + fused_ms:.2f}")
PY
  )"

  echo "Qiskit+cuQuantum Reuse: ${circuit}_n${qubits}"
  echo "Qiskit gate fusion time: ${fused_export_gate_fusion_ms} [ms]"
  echo "cuQuantum simulation time: ${fused_simulation_ms} [ms]"
  echo "Qiskit+cuQuantum total time: ${fused_total_ms} [ms]"
  echo
}

run_suite() {
  local log_path="${LOG_DIR}/qiskit_cuquantum_reuse.txt"
  : > "${log_path}"
  {
    if (( $# > 0 )); then
      if (( $# % 2 != 0 )); then
        echo "[qiskit_cuquantum.sh] Usage: bash qiskit_cuquantum.sh [circuit qubits]..." >&2
        return 1
      fi
      while (( $# > 0 )); do
        print_case "$1" "$2"
        shift 2
      done
    else
      print_case tsp 16
      print_case vqe 12
      print_case vqe 14
      print_case vqe 16
      print_case qv 12
      print_case qv 14
      print_case qaoa 13
      print_case qaoa 15
      print_case qft 14
      print_case qft 16
      print_case qft 18
      print_case portfolio_vqe 16
      print_case portfolio_vqe 17
      print_case portfolio_vqe 18
      print_case graph_state 16
      print_case graph_state 18
    fi
  } | tee "${log_path}"
}

run_suite "$@"
