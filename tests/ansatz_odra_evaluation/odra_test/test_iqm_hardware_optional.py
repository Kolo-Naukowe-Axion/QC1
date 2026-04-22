"""Optional IQM / ODRA hardware checks (set ``IQM_TOKEN``; uses network)."""

from __future__ import annotations

import os

import numpy as np
import pytest
from qiskit import transpile

from odra_test.calibration_metadata import calibration_set_id
from odra_test.expectation_from_counts import counts_to_expectation


@pytest.fixture(scope="module")
def iqm_backend():
    # IQM forbids mixing explicit token=... with IQM_TOKEN in the environment (TokenManager).
    token = os.environ.get("IQM_TOKEN", "").strip()
    if not token:
        pytest.skip(
            "IQM_TOKEN is not set in this process environment. "
            "Export it in the same shell before pytest, e.g. "
            "IQM_TOKEN='<token>' uv run pytest tests/ansatz_odra_evaluation/odra_test -m iqm"
        )
    url = os.environ.get("IQM_URL", "https://odra5.e-science.pl/").strip()
    from iqm.qiskit_iqm import IQMProvider

    provider = IQMProvider(url)
    return provider.get_backend()


@pytest.mark.iqm
def test_iqm_single_job_counts_and_expectation_in_range(iqm_backend):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(5)
    qc.measure_all()
    tqc = transpile(qc, iqm_backend, optimization_level=3)
    job = iqm_backend.run(tqc, shots=256)
    result = job.result()
    counts = result.get_counts()
    if isinstance(counts, list):
        counts = counts[0]
    ev = counts_to_expectation(counts)
    assert -1.0 - 1e-6 <= ev <= 1.0 + 1e-6
    assert sum(counts.values()) == 256


@pytest.mark.iqm
def test_iqm_calibration_metadata_may_be_present(iqm_backend):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(5)
    qc.measure_all()
    tqc = transpile(qc, iqm_backend, optimization_level=3)
    result = iqm_backend.run(tqc, shots=64).result()
    cid = calibration_set_id(result)
    assert cid is None or isinstance(cid, str)


@pytest.mark.iqm
def test_iqm_repeated_identical_jobs_show_shot_noise(iqm_backend):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(5)
    qc.ry(np.pi / 3, 0)
    qc.measure_all()
    tqc = transpile(qc, iqm_backend, optimization_level=3)
    shots = 512
    n_rep = 12
    evs = []
    for _ in range(n_rep):
        result = iqm_backend.run(tqc, shots=shots).result()
        cts = result.get_counts()
        if isinstance(cts, list):
            cts = cts[0]
        evs.append(counts_to_expectation(cts))
    evs = np.array(evs)
    assert float(np.std(evs, ddof=1)) > 1e-4


@pytest.mark.iqm
def test_iqm_three_circuits_in_one_job_return_three_outcomes(iqm_backend):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(5)
    qc.ry(0.7, 2)
    qc.measure_all()
    tqc = transpile(qc, iqm_backend, optimization_level=3)
    shots = 256
    batch_res = iqm_backend.run([tqc, tqc, tqc], shots=shots).result()
    batch_counts = batch_res.get_counts()
    assert isinstance(batch_counts, list) and len(batch_counts) == 3
    for c in batch_counts:
        assert sum(c.values()) == shots
        assert -1.0 - 1e-6 <= counts_to_expectation(c) <= 1.0 + 1e-6
