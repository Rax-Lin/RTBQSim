#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QISKIT_DIR="${ROOT_DIR}/qiskit_test"
LOG_DIR="${ROOT_DIR}/log"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

: "${QISKIT_ROUNDS:=50}"
: "${QISKIT_CPU_THREADS:=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 0)}"
: "${QISKIT_FUSION_THRESHOLD:=5}"
: "${QISKIT_FUSION_MAX_QUBIT:=2}"
: "${QISKIT_FUSION_ENGINE:=transpiler}"
: "${QISKIT_FUSION_DEVICE:=gpu}"
: "${QISKIT_PYTHON_BIN:=}"

mkdir -p "${LOG_DIR}"
mkdir -p "${ROOT_DIR}/log/results/state"
mkdir -p "${ROOT_DIR}/log/fused_gates"

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
  local -a required_mods=("qiskit" "qiskit_aer" "numpy")
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
  echo "[qiskit.sh] No usable Python interpreter with qiskit, qiskit_aer, and numpy was found." >&2
  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    echo "[qiskit.sh] Install them into the local venv with:" >&2
    echo "  ${VENV_DIR}/bin/python -m pip install --upgrade pip" >&2
    echo "  ${VENV_DIR}/bin/python -m pip install qiskit numpy qiskit-aer" >&2
  else
    echo "[qiskit.sh] Create a local venv and install them with:" >&2
    echo "  python3 -m venv ${VENV_DIR}" >&2
    echo "  ${VENV_DIR}/bin/python -m pip install --upgrade pip" >&2
    echo "  ${VENV_DIR}/bin/python -m pip install qiskit numpy qiskit-aer" >&2
  fi
  echo "[qiskit.sh] You can also override the interpreter with QISKIT_PYTHON_BIN=/path/to/python." >&2
  exit 1
fi

run_case() {
  local device="$1"
  local fusion="$2"
  local circuit="$3"
  local qubits="$4"
  "${PYTHON_BIN}" "${QISKIT_DIR}/qiskit_test.py" \
    --circuit_name "${circuit}" \
    --num_qubits "${qubits}" \
    --rounds "${QISKIT_ROUNDS}" \
    --device "${device}" \
    --fusion "${fusion}" \
    --fusion-threshold "${QISKIT_FUSION_THRESHOLD}" \
    --fusion-max-qubit "${QISKIT_FUSION_MAX_QUBIT}" \
    --max-parallel-threads "${QISKIT_CPU_THREADS}"
}

run_export_case() {
  local circuit="$1"
  local qubits="$2"
  "${PYTHON_BIN}" "${QISKIT_DIR}/qiskit_export_gate.py" \
    --circuit_name "${circuit}" \
    --num_qubits "${qubits}" \
    --fusion-engine "${QISKIT_FUSION_ENGINE}" \
    --device "${QISKIT_FUSION_DEVICE}" \
    --fusion-threshold "${QISKIT_FUSION_THRESHOLD}" \
    --fusion-max-qubit "${QISKIT_FUSION_MAX_QUBIT}" \
    --max-parallel-threads "${QISKIT_CPU_THREADS}" \
    --output-basename "qiskit_${circuit}_n${qubits}.txt"
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

print_total_case() {
  local device="$1"
  local circuit="$2"
  local qubits="$3"
  local runtime_output
  local runtime_ms

  runtime_output="$(run_case "${device}" 0 "${circuit}" "${qubits}")"
  runtime_ms="$(extract_ms "Qiskit runtime" "${runtime_output}")"
  if [[ -z "${runtime_ms}" ]]; then
    echo "[qiskit.sh] Failed to parse Qiskit runtime for ${circuit}_n${qubits} [${device}_no_fusion]." >&2
    return 1
  fi

  echo "Qiskit Total: ${circuit}_n${qubits} [${device}_no_fusion]"
  echo "Qiskit total time: ${runtime_ms} [ms]"
  echo
}

print_fusion_export_case() {
  local circuit="$1"
  local qubits="$2"
  local export_output
  local fusion_ms
  local fused_file
  local original_ops
  local fused_ops

  export_output="$(run_export_case "${circuit}" "${qubits}")"
  fusion_ms="$(extract_ms "Qiskit pure fusion time" "${export_output}")"
  fused_file="$(extract_value "Fused gate file" "${export_output}")"
  original_ops="$(extract_value "Original ops" "${export_output}")"
  fused_ops="$(extract_value "Fused output ops" "${export_output}")"
  if [[ -z "${fusion_ms}" ]]; then
    echo "[qiskit.sh] Failed to parse pure fusion time for ${circuit}_n${qubits} [gpu_fusion_export]." >&2
    return 1
  fi
  if [[ -z "${fused_file}" ]]; then
    echo "[qiskit.sh] Failed to parse fused gate file path for ${circuit}_n${qubits} [gpu_fusion_export]." >&2
    return 1
  fi

  echo "Qiskit Fusion Export: ${circuit}_n${qubits} [${QISKIT_FUSION_ENGINE}]"
  echo "Qiskit pure fusion time: ${fusion_ms} [ms]"
  if [[ -n "${original_ops}" && -n "${fused_ops}" ]]; then
    echo "Gate count: ${original_ops} -> ${fused_ops}"
  fi
  echo "Fused gate file: ${fused_file}"
  echo
}

run_no_fusion_suite() {
  local device="$1"
  local label="$2"
  local log_path="${LOG_DIR}/qiskit_${label}.txt"
  : > "${log_path}"
  {
    print_total_case "${device}" tsp 16
    print_total_case "${device}" vqe 12
    print_total_case "${device}" vqe 14
    print_total_case "${device}" vqe 16
    print_total_case "${device}" qv 12
    print_total_case "${device}" qv 14
    print_total_case "${device}" qaoa 13
    print_total_case "${device}" qaoa 15
    print_total_case "${device}" qft 14
    print_total_case "${device}" qft 16
    print_total_case "${device}" qft 18
    print_total_case "${device}" portfolio_vqe 16
    print_total_case "${device}" portfolio_vqe 17
    print_total_case "${device}" portfolio_vqe 18
    print_total_case "${device}" graph_state 16
    print_total_case "${device}" graph_state 18
    print_total_case "${device}" graph_state 20
    print_total_case "${device}" dnn 17
    print_total_case "${device}" dnn 19
    print_total_case "${device}" dnn 21
  } | tee "${log_path}"
}

run_fusion_export_suite() {
  local log_path="${LOG_DIR}/qiskit_${QISKIT_FUSION_ENGINE}_fusion_export.txt"
  : > "${log_path}"
  {
    print_fusion_export_case tsp 16
    print_fusion_export_case vqe 12
    print_fusion_export_case vqe 14
    print_fusion_export_case vqe 16
    print_fusion_export_case qv 12
    print_fusion_export_case qv 14
    print_fusion_export_case qaoa 13
    print_fusion_export_case qaoa 15
    print_fusion_export_case qft 14
    print_fusion_export_case qft 16
    print_fusion_export_case qft 18
    print_fusion_export_case portfolio_vqe 16
    print_fusion_export_case portfolio_vqe 17
    print_fusion_export_case portfolio_vqe 18
    print_fusion_export_case graph_state 16
    print_fusion_export_case graph_state 18
    print_fusion_export_case graph_state 20
    print_fusion_export_case dnn 17
    print_fusion_export_case dnn 19
    print_fusion_export_case dnn 21
  } | tee "${log_path}"
}

run_no_fusion_suite gpu "gpu_no_fusion"
run_no_fusion_suite cpu "cpu_no_fusion"
run_fusion_export_suite
