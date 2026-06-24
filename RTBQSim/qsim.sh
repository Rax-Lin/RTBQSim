#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QSIM_DIR="${ROOT_DIR}/qsim_test"
LOG_DIR="${ROOT_DIR}/log"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

PYTHON_BIN="${VENV_DIR}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

: "${QSIM_ROUNDS:=50}"
: "${QSIM_CPU_THREADS:=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}"
: "${QSIM_MAX_FUSED_GATE_SIZE:=2}"
: "${QSIM_GPU_MODE:=0}"

mkdir -p "${LOG_DIR}"
mkdir -p "${ROOT_DIR}/log/results/state"

"${PYTHON_BIN}" - <<'PY'
import importlib
mods = ("qiskit", "cirq", "qsimcirq", "numpy")
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    raise SystemExit("Missing Python modules: " + ", ".join(missing))
PY

run_case() {
  local device="$1"
  local circuit="$2"
  local qubits="$3"
  "${PYTHON_BIN}" "${QSIM_DIR}/qsim_test.py" \
    --circuit_name "${circuit}" \
    --num_qubits "${qubits}" \
    --rounds "${QSIM_ROUNDS}" \
    --device "${device}" \
    --max-fused-gate-size "${QSIM_MAX_FUSED_GATE_SIZE}" \
    --cpu-threads "${QSIM_CPU_THREADS}" \
    --gpu-mode "${QSIM_GPU_MODE}"
}

gpu_supported() {
  "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import qsimcirq
qsimcirq.QSimSimulator(
    qsim_options=qsimcirq.QSimOptions(
        max_fused_gate_size=2,
        use_gpu=True,
        gpu_mode=0,
        verbosity=0,
    )
)
PY
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

print_total_case() {
  local device="$1"
  local circuit="$2"
  local qubits="$3"
  local runtime_output
  local runtime_ms

  runtime_output="$(run_case "${device}" "${circuit}" "${qubits}")"
  runtime_ms="$(extract_ms "Qsim runtime" "${runtime_output}")"
  if [[ -z "${runtime_ms}" ]]; then
    echo "[qsim.sh] Failed to parse Qsim runtime for ${circuit}_n${qubits} [${device}]." >&2
    return 1
  fi

  echo "Qsim Total: ${circuit}_n${qubits} [${device}]"
  echo "Qsim total time: ${runtime_ms} [ms]"
  echo
}

run_suite() {
  local device="$1"
  local log_path="${LOG_DIR}/qsim_${device}.txt"
  : > "${log_path}"
  if [[ "${device}" == "gpu" ]] && ! gpu_supported; then
    {
      echo "[qsim.sh] GPU qsim is not available in the current Python environment."
      echo "[qsim.sh] Install/compile qsimcirq with GPU support to enable GPU baseline runs."
    } | tee "${log_path}"
    return 0
  fi
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
  } | tee "${log_path}"
}

run_suite gpu
run_suite cpu
