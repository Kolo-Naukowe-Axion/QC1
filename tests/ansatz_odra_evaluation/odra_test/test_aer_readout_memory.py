"""Aer: aggregated shot memory matches ``get_counts``; readout mapping on |0…0⟩."""

from collections import Counter

import pytest
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from odra_test.expectation_from_counts import counts_to_expectation


def _memory_to_counts(memory: list[str]) -> dict[str, int]:
    return dict(Counter(memory))


@pytest.fixture
def aer_backend():
    return AerSimulator()


def test_memory_aggregates_to_counts(aer_backend):
    qc = QuantumCircuit(5)
    qc.h(0)
    for i in range(4):
        qc.cx(i, i + 1)
    qc.measure_all()
    shots = 2048
    job = aer_backend.run(qc, shots=shots, memory=True)
    result = job.result()
    mem = result.get_memory()
    from_mem = _memory_to_counts(mem)
    from_res = result.get_counts()
    assert sum(from_mem.values()) == shots
    assert from_mem == from_res


def test_expectation_all_zeros_statevector_limit(aer_backend):
    qc = QuantumCircuit(5)
    qc.measure_all()
    result = aer_backend.run(qc, shots=4000, memory=True).result()
    ev = counts_to_expectation(result.get_counts())
    assert ev == pytest.approx(1.0, abs=0.05)
