#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import cirq
import numpy as np
import qiskit
import qiskit.qasm2
import qsimcirq
from qiskit.quantum_info import Operator


def read_complex_numbers(filename: Path) -> np.ndarray:
    with open(filename, "r", encoding="utf-8") as file:
        line = file.readline()

    numbers = list(map(float, line.split()))
    complex_numbers = []
    for i in range(0, len(numbers), 2):
        complex_numbers.append(complex(numbers[i], numbers[i + 1]))
    return np.array(complex_numbers, dtype=np.complex128)


def write_complex_numbers(filename: Path, complex_array: np.ndarray) -> None:
    with open(filename, "w", encoding="utf-8") as file:
        for cnum in complex_array:
            file.write(f"{cnum.real} {cnum.imag}\n")


def bit_reverse_permute(state: np.ndarray, num_qubits: int) -> np.ndarray:
    permuted = np.empty_like(state)
    for idx in range(state.shape[0]):
        reversed_idx = 0
        value = idx
        for _ in range(num_qubits):
            reversed_idx = (reversed_idx << 1) | (value & 1)
            value >>= 1
        permuted[reversed_idx] = state[idx]
    return permuted


def load_qiskit_circuit(circuit_path: Path):
    try:
        return qiskit.qasm2.load(str(circuit_path))
    except Exception:
        return qiskit.qasm2.load(
            str(circuit_path),
            custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
        )


def qiskit_to_cirq_circuit(qiskit_circuit) -> cirq.Circuit:
    qubits = cirq.LineQubit.range(qiskit_circuit.num_qubits)
    cirq_circuit = cirq.Circuit()

    for inst in qiskit_circuit.data:
        op = inst.operation
        name = op.name.lower()

        if name == "barrier":
            continue
        if name in {"measure", "reset", "initialize"}:
            raise ValueError(f"Unsupported non-unitary operation for qsim baseline: {name}")
        qargs = [qiskit_circuit.find_bit(qubit).index for qubit in inst.qubits]
        # Qiskit and Cirq use different qubit-order conventions for matrix application.
        cirq_qubits = [qubits[i] for i in reversed(qargs)]
        mat = np.asarray(Operator(op).data, dtype=np.complex128)
        cirq_circuit.append(cirq.MatrixGate(mat).on(*cirq_qubits))

    return cirq_circuit


def build_simulator(device: str, max_fused_gate_size: int, cpu_threads: int, gpu_mode: int):
    options = qsimcirq.QSimOptions(
        max_fused_gate_size=max_fused_gate_size,
        cpu_threads=max(1, cpu_threads),
        use_gpu=(device.lower() == "gpu"),
        gpu_mode=gpu_mode,
        verbosity=0,
    )
    return qsimcirq.QSimSimulator(qsim_options=options)


def main() -> int:
    parser = argparse.ArgumentParser(description="qsim baseline runner.")
    parser.add_argument("--circuit_name", "-c", type=str, required=True,
                        help="Name of the circuit (e.g., dnn, vqe)")
    parser.add_argument("--num_qubits", "-n", type=int, required=True,
                        help="Number of qubits")
    parser.add_argument("--rounds", "-r", type=int, default=1,
                        help="How many times to rerun the same circuit")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu",
                        help="qsim execution device")
    parser.add_argument("--max-fused-gate-size", type=int, default=2,
                        help="qsim max_fused_gate_size")
    parser.add_argument("--cpu-threads", type=int, default=1,
                        help="CPU-only: qsim cpu_threads")
    parser.add_argument("--gpu-mode", type=int, default=0,
                        help="qsim gpu_mode (0=CUDA, 1=cuStateVec, >=2=cuStateVecEx)")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Optional suffix appended to output state filename")
    args = parser.parse_args()

    circuit_name = args.circuit_name
    num_qubits = args.num_qubits
    rounds = args.rounds
    device = args.device.lower()
    tag = device

    base_dir = Path(__file__).resolve().parent.parent
    circuit_path = base_dir / "circuits" / f"{circuit_name}_n{num_qubits}.qasm"
    input_path = base_dir / "input_batch" / f"n{num_qubits}.txt"
    output_dir = base_dir / "log" / "results" / "state"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    output_path = output_dir / f"qsim_{tag}_{circuit_name}_n{num_qubits}{suffix}.txt"

    input_state_np = read_complex_numbers(input_path).astype(np.complex64)
    qsim_input_state = bit_reverse_permute(input_state_np, num_qubits)
    qiskit_circuit = load_qiskit_circuit(circuit_path)
    cirq_circuit = qiskit_to_cirq_circuit(qiskit_circuit)
    simulator = build_simulator(
        device=device,
        max_fused_gate_size=args.max_fused_gate_size,
        cpu_threads=args.cpu_threads,
        gpu_mode=args.gpu_mode,
    )

    result = None
    begin = time.perf_counter()
    try:
        for _ in range(rounds):
            result = simulator.simulate(cirq_circuit, initial_state=qsim_input_state)
    except Exception as exc:
        raise SystemExit(f"qsim simulation failed on device={device}: {exc}") from exc
    end = time.perf_counter()
    runtime_ms = (end - begin) * 1000.0

    if result is None:
        raise SystemExit("qsim simulation produced no result.")

    final_state = np.asarray(result.final_state_vector, dtype=np.complex128)
    final_state = bit_reverse_permute(final_state, num_qubits)
    write_complex_numbers(output_path, final_state)

    print(f"Qsim Baseline: {circuit_name}_n{num_qubits} [{tag}]")
    print(f"Qsim runtime: {runtime_ms:.2f} [ms]")
    print(f"Rounds: {rounds}")
    print(f"Max fused gate size: {args.max_fused_gate_size}")
    if device == "cpu":
        print(f"CPU threads: {args.cpu_threads}")
    else:
        print(f"GPU mode: {args.gpu_mode}")
    print(f"Output saved: {output_path}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
