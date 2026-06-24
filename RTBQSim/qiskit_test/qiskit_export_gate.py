#!/usr/bin/python3
import argparse
import time
from pathlib import Path

import qiskit
import qiskit.qasm2
from qiskit import QuantumCircuit
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks


def load_circuit(circuit_path: Path):
    try:
        return qiskit.qasm2.load(str(circuit_path))
    except Exception:
        return qiskit.qasm2.load(
            str(circuit_path),
            custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
        )


def build_fusion_pass_manager():
    return PassManager([
        Collect2qBlocks(),
        ConsolidateBlocks(force_consolidate=True),
    ])


def materialize_transpiler_fused_ops(circuit):
    fused_ops = []
    for inst in circuit.data:
        op = inst.operation
        qubits = [circuit.find_bit(qubit).index for qubit in inst.qubits]
        mat = Operator(op).data
        fused_ops.append({
            "name": op.name,
            "qubits": qubits,
            "matrix": mat,
        })
    return fused_ops


def materialize_raw_ops(circuit):
    raw_ops = []
    for inst in circuit.data:
        op = inst.operation
        name = op.name.lower()
        if name in {"barrier", "measure", "reset", "initialize"}:
            continue
        qubits = [circuit.find_bit(qubit).index for qubit in inst.qubits]
        mat = Operator(op).data
        raw_ops.append({
            "name": op.name,
            "qubits": qubits,
            "matrix": mat,
        })
    return raw_ops


def build_aer_simulator(device, fusion_threshold, fusion_max_qubit, max_parallel_threads):
    try:
        from qiskit_aer import AerSimulator
    except ImportError as exc:
        raise RuntimeError(
            "qiskit_aer is required only for --fusion-engine aer. "
            "Install qiskit-aer or switch to --fusion-engine transpiler."
        ) from exc

    options = {
        "method": "statevector",
        "device": device.upper(),
        "fusion_enable": True,
        "fusion_verbose": True,
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


def filtered_aer_output_ops(fusion_meta):
    ops = fusion_meta.get("output_ops", [])
    result = []
    ignored_names = {
        "set_statevector",
        "initialize",
        "reset",
        "barrier",
    }
    for op in ops:
        name = op.get("name", "")
        if name.startswith("save_") or name in ignored_names:
            continue
        result.append(op)
    return result


def matrix_from_aer_gate(gate):
    gate_name = gate["name"]
    qubit_count = len(gate["qubits"])
    dim = 2 ** qubit_count

    if gate_name == "unitary":
        raw = gate["mats"][0]
        mat = []
        for i in range(dim):
            row = []
            for j in range(dim):
                value = raw[i][j]
                row.append(complex(float(value[0]), float(value[1])))
            mat.append(row)
        return mat

    if gate_name == "diagonal":
        params = gate["params"]
        mat = []
        for i in range(dim):
            row = []
            for j in range(dim):
                if i == j:
                    value = params[i]
                    row.append(complex(float(value[0]), float(value[1])))
                else:
                    row.append(0.0 + 0.0j)
            mat.append(row)
        return mat

    standard_gates = get_standard_gate_name_mapping()
    if gate_name not in standard_gates:
        raise ValueError(f"Unsupported fused gate name for export: {gate_name}")

    template_gate = standard_gates[gate_name]
    params = gate.get("params", [])
    scalar_params = []
    for param in params:
        if isinstance(param, (list, tuple)) and len(param) == 2:
            scalar_params.append(float(param[0]))
        else:
            scalar_params.append(float(param))
    gate_obj = template_gate.__class__(*scalar_params) if scalar_params else template_gate
    return Operator(gate_obj).data


def materialize_aer_fused_ops(circuit, device, fusion_threshold, fusion_max_qubit, max_parallel_threads):
    prep = QuantumCircuit(circuit.num_qubits)
    fused_input = prep.compose(circuit)
    fused_input.save_statevector()

    sim = build_aer_simulator(device, fusion_threshold, fusion_max_qubit, max_parallel_threads)
    result = sim.run(fused_input).result()
    metadata = result.results[0].metadata
    fusion_meta = metadata.get("fusion", {})
    output_ops = filtered_aer_output_ops(fusion_meta)

    fused_ops = []
    for gate in output_ops:
        fused_ops.append({
            "name": gate["name"],
            "qubits": list(gate["qubits"]),
            "matrix": matrix_from_aer_gate(gate),
        })
    fusion_pass_ms = float(fusion_meta.get("time_taken", 0.0)) * 1000.0
    return fused_ops, fusion_pass_ms


def write_fused_ops(file_path: Path, fused_ops):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(str(len(fused_ops)) + "\n")
        for gate in fused_ops:
            qubits = gate["qubits"]
            dim = 2 ** len(qubits)
            tensor_size = dim * dim

            file.write(str(len(qubits)) + "\n")
            file.write(" ".join(str(q) for q in qubits) + "\n")
            file.write(str(tensor_size) + "\n")

            mat = gate["matrix"]
            for i in range(dim):
                for j in range(dim):
                    value = mat[i][j]
                    file.write(f"{float(value.real)} {float(value.imag)} ")
            file.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Qiskit fused-gate exporter.")
    parser.add_argument("--circuit_name", "-c", type=str, required=True,
                        help="Name of the circuit (e.g., dnn, vqe)")
    parser.add_argument("--num_qubits", "-n", type=int, required=True,
                        help="Number of qubits")
    parser.add_argument("--fusion-engine", choices=("raw", "transpiler", "aer"), default="transpiler",
                        help="Gate export mode: raw circuit gates, transpiler fusion, or Aer runtime fusion")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu",
                        help="Aer-only: AerSimulator device")
    parser.add_argument("--fusion-threshold", type=int, default=5,
                        help="Aer-only: Qiskit Aer fusion_threshold")
    parser.add_argument("--fusion-max-qubit", type=int, default=2,
                        help="Aer-only: Qiskit Aer fusion_max_qubit")
    parser.add_argument("--max-parallel-threads", type=int, default=0,
                        help="Aer-only CPU setting: max OpenMP threads (0 means auto)")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Optional suffix appended to fused-gate filename")
    parser.add_argument("--output-basename", type=str, default="",
                        help="Optional explicit fused-gate output basename (without directory)")
    args = parser.parse_args()

    circuit_name = args.circuit_name
    num_qubits = args.num_qubits

    base_dir = Path(__file__).resolve().parent.parent
    circuit_path = base_dir / "circuits" / f"{circuit_name}_n{num_qubits}.qasm"
    fused_dir = base_dir / "log" / "fused_gates"
    fused_dir.mkdir(parents=True, exist_ok=True)
    if args.output_basename:
        fused_filename = args.output_basename
        if not fused_filename.endswith(".txt"):
            fused_filename += ".txt"
    else:
        suffix = f"_{args.output_suffix}" if args.output_suffix else ""
        fused_filename = f"qiskit_{circuit_name}_n{num_qubits}{suffix}.txt"
    fused_path = fused_dir / fused_filename

    load_begin = time.perf_counter()
    circuit = load_circuit(circuit_path)
    load_end = time.perf_counter()
    fusion_pass_ms = 0.0
    fusion_begin = time.perf_counter()

    if args.fusion_engine == "raw":
        fused_ops = materialize_raw_ops(circuit)
    elif args.fusion_engine == "transpiler":
        pass_manager = build_fusion_pass_manager()
        fused_circuit = pass_manager.run(circuit)
        fused_ops = materialize_transpiler_fused_ops(fused_circuit)
    else:
        fused_ops, fusion_pass_ms = materialize_aer_fused_ops(
            circuit,
            args.device,
            args.fusion_threshold,
            args.fusion_max_qubit,
            args.max_parallel_threads,
        )
    fusion_end = time.perf_counter()
    write_begin = time.perf_counter()
    write_fused_ops(fused_path, fused_ops)
    write_end = time.perf_counter()

    circuit_load_ms = (load_end - load_begin) * 1000.0
    gate_fusion_ms = (fusion_end - fusion_begin) * 1000.0
    write_ms = (write_end - write_begin) * 1000.0
    pure_fusion_ms = (write_end - load_begin) * 1000.0

    print(f"Qiskit Fusion Export: {circuit_name}_n{num_qubits} [{args.fusion_engine}]")
    if args.fusion_engine == "raw":
        print(f"Qiskit circuit load time: {circuit_load_ms:.2f} [ms]")
        print(f"Qiskit raw gate build time: {gate_fusion_ms:.2f} [ms]")
        print(f"Qiskit fused gate write time: {write_ms:.2f} [ms]")
        print(f"Qiskit raw export time: {pure_fusion_ms:.2f} [ms]")
    else:
        print(f"Qiskit circuit load time: {circuit_load_ms:.2f} [ms]")
        print(f"Qiskit gate fusion time: {gate_fusion_ms:.2f} [ms]")
        print(f"Qiskit fused gate write time: {write_ms:.2f} [ms]")
        print(f"Qiskit pure fusion time: {pure_fusion_ms:.2f} [ms]")
    if args.fusion_engine == "aer":
        print(f"Qiskit Aer fusion pass time: {fusion_pass_ms:.2f} [ms]")
    print(f"Fused output ops: {len(fused_ops)}")
    print(f"Fused gate file: {fused_path}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
