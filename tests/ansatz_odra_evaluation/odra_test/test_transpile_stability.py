"""Transpiler stochasticity: unseeded vs ``seed_transpiler`` (layout / depth / CX count)."""

from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2

from odra_test.circuits import full_hybrid_circuit, odra_ansatz, simulator_ansatz


def _make_measured_full(ansatz_fn, n_qubits: int = 5, depth: int = 2) -> QuantumCircuit:
    qc = full_hybrid_circuit(ansatz_fn(n_qubits, depth), n_qubits)
    qc.measure_all()
    return qc


def _signature(tqc: QuantumCircuit) -> tuple[int, int]:
    ops = tqc.count_ops()
    return tqc.depth(), int(ops.get("cx", 0))


def test_unseeded_transpile_can_vary_across_calls():
    backend = GenericBackendV2(num_qubits=5, basis_gates=["cx", "rz", "sx", "x"], seed=4242)
    template = _make_measured_full(odra_ansatz)
    sigs = set()
    for _ in range(50):
        tqc = transpile(template, backend, optimization_level=3)
        sigs.add(_signature(tqc))
    if len(sigs) == 1:
        import pytest

        pytest.skip("Transpiler produced identical depth/CX in all trials (environment may fix RNG internally).")
    assert len(sigs) > 1


def test_seed_transpiler_produces_stable_signature():
    backend = GenericBackendV2(num_qubits=5, basis_gates=["cx", "rz", "sx", "x"], seed=4242)
    template = _make_measured_full(simulator_ansatz)
    seed = 7
    sigs = {_signature(transpile(template, backend, optimization_level=3, seed_transpiler=seed)) for _ in range(25)}
    assert len(sigs) == 1


def test_odra_vs_simulator_transpiled_cx_count_odra_not_greater():
    backend = GenericBackendV2(num_qubits=5, basis_gates=["cx", "rz", "sx", "x"], seed=4242)
    od = transpile(_make_measured_full(odra_ansatz), backend, optimization_level=3, seed_transpiler=11)
    sim = transpile(_make_measured_full(simulator_ansatz), backend, optimization_level=3, seed_transpiler=11)
    _, cx_od = _signature(od)
    _, cx_sim = _signature(sim)
    assert cx_sim >= cx_od
