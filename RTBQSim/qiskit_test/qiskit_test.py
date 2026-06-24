#!/usr/bin/python3
import argparse
import time
from pathlib import Path

import numpy as np
import qiskit
import qiskit.qasm2
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def read_complex_numbers(filename):
    with open(filename, "r", encoding="utf-8") as file:
        line = file.readline()

    complex_numbers = []
    numbers = list(map(float, line.split()))
    for i in range(0, len(numbers), 2):
        real = numbers[i]
        imag = numbers[i + 1]
        complex_numbers.append(complex(real, imag))

    return np.array(complex_numbers, dtype=np.complex128)


def write_complex_numbers(filename, complex_array):
    with open(filename, "w", encoding="utf-8") as file:
        for cnum in complex_array:
            file.write(f"{cnum.real} {cnum.imag}\n")


def config_tag(device, fusion_enabled):
    return f"{device.lower()}_{'fusion' if fusion_enabled else 'no_fusion'}"


def build_simulator(device, fusion_enabled, fusion_threshold, fusion_max_qubit, max_parallel_threads):
    options = {
        "method": "statevector",
        "device": device.upper(),
        "fusion_enable": fusion_enabled,
        "fusion_verbose": fusion_enabled,
        "fusion_threshold": fusion_threshold,
        "fusion_max_qubit": fusion_max_qubit,
    }
    if device.lower() == "gpu":
        options["batched_shots_gpu"] = True
    else:
        options["max_parallel_threads"] = max_parallel_threads
        options["max_parallel_experiments"] = 1
        options["statevector_parallel_threshold"] = 1
    return AerSimulator(**options)


def main():
    parser = argparse.ArgumentParser(description="Qiskit Aer baseline runner.")
    parser.add_argument("--circuit_name", "-c", type=str, required=True,
                        help="Name of the circuit (e.g., dnn, vqe)")
    parser.add_argument("--num_qubits", "-n", type=int, required=True,
                        help="Number of qubits")
    parser.add_argument("--rounds", "-r", type=int, default=1,
                        help="How many times to rerun the same circuit")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu",
                        help="AerSimulator device")
    parser.add_argument("--fusion", type=int, choices=(0, 1), default=1,
                        help="Enable Qiskit Aer fusion optimization")
    parser.add_argument("--fusion-threshold", type=int, default=5,
                        help="Qiskit Aer fusion_threshold")
    parser.add_argument("--fusion-max-qubit", type=int, default=5,
                        help="Qiskit Aer fusion_max_qubit")
    parser.add_argument("--max-parallel-threads", type=int, default=0,
                        help="CPU-only: max OpenMP threads (0 means auto)")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Optional suffix appended to output state filename")
    args = parser.parse_args()

    circuit_name = args.circuit_name
    num_qubits = args.num_qubits
    rounds = args.rounds
    fusion_enabled = (args.fusion == 1)
    tag = config_tag(args.device, fusion_enabled)

    base_dir = Path(__file__).resolve().parent.parent
    circuit_path = base_dir / "circuits" / f"{circuit_name}_n{num_qubits}.qasm"
    input_path = base_dir / "input_batch" / f"n{num_qubits}.txt"
    output_dir = base_dir / "log" / "results" / "state"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    output_path = output_dir / f"qiskit_{tag}_{circuit_name}_n{num_qubits}{suffix}.txt"

    input_state_np = read_complex_numbers(input_path)
    input_state_q = qiskit.quantum_info.Statevector(input_state_np)

    circuit1 = QuantumCircuit(num_qubits)
    circuit1.set_statevector(input_state_q)
    try:
        circuit2 = qiskit.qasm2.load(str(circuit_path))
    except Exception:
        circuit2 = qiskit.qasm2.load(
            str(circuit_path),
            custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
        )
    circuit = circuit1.compose(circuit2)
    circuit.save_statevector()

    sim = build_simulator(
        device=args.device,
        fusion_enabled=fusion_enabled,
        fusion_threshold=args.fusion_threshold,
        fusion_max_qubit=args.fusion_max_qubit,
        max_parallel_threads=args.max_parallel_threads,
    )

    statevector = None
    fusion_output_ops = 0
    begin = time.perf_counter()
    for _ in range(rounds):
        job = sim.run(circuit)
        result = job.result()
        statevector = result.get_statevector(circuit)
        metadata = result.results[0].metadata
        fusion_meta = metadata.get("fusion", {})
        output_ops = fusion_meta.get("output_ops", [])
        if output_ops:
            fusion_output_ops = max(fusion_output_ops, max(0, len(output_ops) - 1))
    end = time.perf_counter()
    runtime_ms = (end - begin) * 1000.0

    statevector_np = np.asarray(statevector, dtype=np.complex128)
    write_complex_numbers(output_path, statevector_np)

    print(f"Qiskit Baseline: {circuit_name}_n{num_qubits} [{tag}]")
    print(f"Qiskit runtime: {runtime_ms:.2f} [ms]")
    print(f"Rounds: {rounds}")
    if args.device.lower() == "cpu":
        print(f"CPU threads: {args.max_parallel_threads}")
    print(f"Fusion: {1 if fusion_enabled else 0}")
    if fusion_enabled:
        print(f"Fused output ops: {fusion_output_ops}")
    print(f"Output saved: {output_path}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
