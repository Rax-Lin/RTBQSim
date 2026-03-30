import argparse
from pathlib import Path

import numpy as np
import qiskit.qasm2 as qasm2
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def load_statevector(path) :
    data = np.loadtxt(path, dtype=np.float64)

    if data.ndim == 2:
        if data.shape[1] != 2:
            raise ValueError(f"Unsupported 2D shape {data.shape} in {path}")
        vec = data[:, 0] + 1j * data[:, 1]
        return vec.astype(np.complex128)

    if data.ndim == 1:
        if data.size % 2 == 0:
            return (data[0::2] + 1j * data[1::2]).astype(np.complex128)
        return data.astype(np.complex128)

    raise ValueError(f"Unsupported data shape {data.shape} in {path}")


def normalize(vec, name) :
    norm = np.linalg.norm(vec)
    if np.isclose(norm, 0.0):
        raise ValueError(f"{name} statevector has zero norm")
    return vec / norm


def build_qiskit_reference(qasm_path, input_state_path, device) :
    init_state = load_statevector(input_state_path)
    sim = AerSimulator(method="statevector", device=device.upper())
    try:
        qc_main = qasm2.load(str(qasm_path))
    except Exception:
        qc_main = qasm2.load(
            str(qasm_path),
            custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
        )
    qc_main = transpile(qc_main, sim, optimization_level=0)

    qc = QuantumCircuit(qc_main.num_qubits)
    qc.set_statevector(init_state)
    qc = qc.compose(qc_main)
    qc.save_statevector()

    result = sim.run(qc).result()
    return np.asarray(result.get_statevector(qc), dtype=np.complex128)


def verify(
    circuit_name,
    num_qubits,
    qasm_path,
    input_state_path,
    qbsim_path,
    device,
    fidelity_threshold,
    rmse_threshold,
) :
    for p in (qasm_path, input_state_path, qbsim_path):
        if not p.exists():
            print(f"Missing file: {p}")
            return 1
    print(f"=======================================")
    print(f"benchmark:    {circuit_name}_n{num_qubits}")
    # print(f"QASM:  {qasm_path}")
    # print(f"Input: {input_state_path}")
    # print(f"QBSim: {qbsim_path}")

    v_sim = load_statevector(qbsim_path)
    v_ref = build_qiskit_reference(qasm_path, input_state_path, device)

    if len(v_sim) != len(v_ref):
        print(f"Length mismatch: QBSim={len(v_sim)} vs Qiskit={len(v_ref)}")
        return 1

    v_sim = normalize(v_sim, "QBSim")
    v_ref = normalize(v_ref, "Qiskit")

    overlap = np.vdot(v_sim, v_ref)
    if np.abs(overlap) > 0:
        v_sim_aligned = v_sim * np.exp(1j * np.angle(overlap))
    else:
        v_sim_aligned = v_sim

    fidelity = np.clip(np.abs(np.vdot(v_ref, v_sim)) ** 2, 0.0, 1.0)
    rmse = np.sqrt(np.mean(np.abs(v_ref - v_sim_aligned) ** 2))
    max_abs_err = np.max(np.abs(v_ref - v_sim_aligned))

    print(f"Fidelity:     {fidelity:.12f}")
    print(f"RMSE:         {rmse:.6e}")
    print(f"Max abs diff: {max_abs_err:.6e}")

    passed = fidelity >= fidelity_threshold and rmse <= rmse_threshold
    if passed:
        print("PASS")
        return 0

    print(
        f"FAIL (thresholds: fidelity>={fidelity_threshold}, rmse<={rmse_threshold})"
    )
    return 2


def main() :
    parser = argparse.ArgumentParser(description="Verify QBSim statevector with Qiskit.")
    parser.add_argument("-c", "--circuit", type=str, required=True)
    parser.add_argument("-n", "--qubits", type=int, required=True)
    parser.add_argument(
        "--device",
        type=str,
        choices=("cpu", "gpu"),
        default="cpu",
        help="Qiskit Aer backend device.",
    )
    parser.add_argument("--fidelity-threshold", type=float, default=0.9999)
    parser.add_argument("--rmse-threshold", type=float, default=1e-5)
    parser.add_argument("--qasm", type=str, default=None)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--qbsim", type=str, default=None)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    qasm_path = Path(args.qasm) if args.qasm else base / "circuits" / f"{args.circuit}_n{args.qubits}.qasm"
    input_path = Path(args.input) if args.input else base / "input_batch" / f"n{args.qubits}.txt"
    qbsim_path = (
        Path(args.qbsim)
        if args.qbsim
        else base / "log" / "results" / "state" / f"qbsim_{args.circuit}_n{args.qubits}.txt"
    )

    return verify(
        circuit_name=args.circuit,
        num_qubits=args.qubits,
        qasm_path=qasm_path,
        input_state_path=input_path,
        qbsim_path=qbsim_path,
        device=args.device,
        fidelity_threshold=args.fidelity_threshold,
        rmse_threshold=args.rmse_threshold,
    )


if __name__ == "__main__":
    raise SystemExit(main())
