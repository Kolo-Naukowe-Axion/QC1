"""``IQMBackendEstimator`` + fake ``backend.run`` (binomial shots) matches binomial scaling."""

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from odra_test.circuits import full_hybrid_circuit, odra_ansatz
from odra_test.fake_shot_job import fake_run_binomial_last_bit
from odra_test.iqm_backend_estimator import IQMBackendEstimator


def _pub_single_row(n_qubits: int = 5):
    ans = odra_ansatz(n_qubits, 2)
    qc = full_hybrid_circuit(ans, n_qubits)
    obs = SparsePauliOp.from_list([("I" * (n_qubits - 1) + "Z", 1.0)])
    x = np.random.default_rng(0).uniform(-0.5, 0.5, size=(len(qc.parameters),))
    return qc, obs, x


def test_fake_backend_single_row_expectation_variance():
    from qiskit.providers.fake_provider import GenericBackendV2

    n_qubits = 5
    shots = 512
    p = 0.55
    rng = np.random.default_rng(99)
    backend = GenericBackendV2(num_qubits=n_qubits, basis_gates=["cx", "rz", "sx", "x"], seed=1)
    qc, obs, x = _pub_single_row(n_qubits)
    est = IQMBackendEstimator(
        backend,
        options={"shots": shots, "seed_transpiler": 42, "optimization_level": 3},
    )
    backend.run = fake_run_binomial_last_bit(n_qubits, shots, rng, p)

    n_runs = 2500
    out = []
    for _ in range(n_runs):
        job = est.run([(qc, obs, x)])
        ev = job.result()[0].data.evs
        assert ev.shape == (1,)
        out.append(float(ev[0]))
    out = np.array(out)
    emp = float(np.var(out, ddof=1))
    theory = 4.0 * p * (1.0 - p) / shots
    ratio = emp / theory
    assert 0.75 < ratio < 1.25


def test_fake_backend_batch_matches_row_count():
    from qiskit.providers.fake_provider import GenericBackendV2

    n_qubits = 5
    shots = 256
    rng = np.random.default_rng(3)
    backend = GenericBackendV2(num_qubits=n_qubits, basis_gates=["cx", "rz", "sx", "x"], seed=2)
    qc, obs, _ = _pub_single_row(n_qubits)
    grid = np.random.default_rng(4).uniform(-1, 1, size=(7, len(qc.parameters)))
    est = IQMBackendEstimator(
        backend,
        options={"shots": shots, "seed_transpiler": 0, "optimization_level": 3},
    )
    backend.run = fake_run_binomial_last_bit(n_qubits, shots, rng, 0.5)
    job = est.run([(qc, obs, grid)])
    evs = job.result()[0].data.evs
    assert evs.shape == (7,)
    assert np.all(np.abs(evs) <= 1.0 + 1e-9)


def test_max_circuits_per_job_splits_backend_calls():
    from qiskit.providers.fake_provider import GenericBackendV2

    n_qubits = 5
    shots = 128
    rng = np.random.default_rng(5)
    backend = GenericBackendV2(num_qubits=n_qubits, basis_gates=["cx", "rz", "sx", "x"], seed=3)
    qc, obs, _ = _pub_single_row(n_qubits)
    grid = np.random.default_rng(6).uniform(-0.3, 0.3, size=(5, len(qc.parameters)))
    est = IQMBackendEstimator(
        backend,
        options={
            "shots": shots,
            "seed_transpiler": 99,
            "optimization_level": 3,
            "max_circuits_per_job": 2,
        },
    )
    inner = fake_run_binomial_last_bit(n_qubits, shots, rng, 0.5)
    batch_sizes: list[int] = []

    def wrapped(circuits, shots_arg=None, **_kwargs):
        batch_sizes.append(len(circuits))
        return inner(circuits, shots_arg, **_kwargs)

    backend.run = wrapped
    job = est.run([(qc, obs, grid)])
    assert job.result()[0].data.evs.shape == (5,)
    assert batch_sizes == [2, 2, 1]


def test_failed_backend_call_is_tracked():
    from qiskit.providers.fake_provider import GenericBackendV2

    n_qubits = 5
    shots = 128
    backend = GenericBackendV2(num_qubits=n_qubits, basis_gates=["cx", "rz", "sx", "x"], seed=4)
    qc, obs, x = _pub_single_row(n_qubits)
    est = IQMBackendEstimator(
        backend,
        options={"shots": shots, "seed_transpiler": 7, "optimization_level": 1},
    )

    def failing_run(_circuits, _shots=None, **_kwargs):
        raise RuntimeError("simulated backend timeout")

    backend.run = failing_run
    job = est.run([(qc, obs, x)])
    evs = job.result()[0].data.evs

    assert evs.shape == (1,)
    assert float(evs[0]) == 0.0
    assert len(est.failed_batches) == 1
    assert "simulated backend timeout" in est.failed_batches[0]["error"]
