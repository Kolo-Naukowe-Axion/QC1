"""Ansätze aligned with `tests/divergence` checkpoints (20 params, full ring, depth 2)."""

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def angle_encoding(n_qubits: int) -> QuantumCircuit:
    input_params = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(input_params[i], i)
    return qc


def odra_ansatz(n_qubits: int, depth: int) -> QuantumCircuit:
    params_per_iter = 4 * n_qubits
    theta = ParameterVector("theta", params_per_iter * (depth // 2))
    qc = QuantumCircuit(n_qubits)

    for j in range(depth // 2):
        offset = j * params_per_iter

        for i in range(n_qubits):
            qc.ry(theta[offset + i], i)

        for i in range(n_qubits):
            control = i
            target = (i + 1) % n_qubits
            param_idx = offset + n_qubits + i
            qc.rz(theta[param_idx], target)
            qc.cz(control, target)

        offset_l2 = offset + 2 * n_qubits
        for i in range(n_qubits):
            qc.rx(theta[offset_l2 + i], i)

        for i in range(n_qubits):
            control = i
            target = (i - 1) % n_qubits
            param_idx = offset_l2 + n_qubits + i
            qc.ry(theta[param_idx], target)
            qc.cz(control, target)

    return qc


def simulator_ansatz(n_qubits: int, depth: int) -> QuantumCircuit:
    theta = ParameterVector("theta", 2 * n_qubits * depth)
    qc = QuantumCircuit(n_qubits)
    param_idx = 0

    for _ in range(depth // 2):
        for i in range(n_qubits):
            qc.ry(theta[param_idx], i)
            param_idx += 1

        for i in range(n_qubits):
            control = i
            target = (i + 1) % n_qubits
            qc.crx(theta[param_idx], control, target)
            param_idx += 1

        for i in range(n_qubits):
            qc.rx(theta[param_idx], i)
            param_idx += 1

        for i in range(n_qubits):
            control = i
            target = (i - 1) % n_qubits
            qc.cry(theta[param_idx], control, target)
            param_idx += 1

    return qc


def full_hybrid_circuit(ansatz: QuantumCircuit, n_qubits: int) -> QuantumCircuit:
    fm = angle_encoding(n_qubits)
    qc = QuantumCircuit(n_qubits)
    qc.compose(fm, qubits=range(n_qubits), inplace=True)
    qc.compose(ansatz, inplace=True)
    return qc


def bind_random_params(qc: QuantumCircuit, rng, scale: float = 1.0) -> QuantumCircuit:
    values = rng.uniform(-scale, scale, size=len(qc.parameters))
    return qc.assign_parameters(values)
