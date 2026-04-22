"""Fake IQM-style job results for unit tests (binomial shots on last measured bit)."""

from __future__ import annotations

import numpy as np


def counts_for_last_bit(n_qubits: int, shots: int, n_last_bit_zero: int) -> dict[str, int]:
    """Two outcome strings differing only on the last character (LSB / qubit 0 in Qiskit bitstring order)."""
    if n_last_bit_zero < 0 or n_last_bit_zero > shots:
        raise ValueError("invalid n_last_bit_zero")
    s0 = "0" * n_qubits
    s1 = "0" * (n_qubits - 1) + "1"
    return {s0: int(n_last_bit_zero), s1: int(shots - n_last_bit_zero)}


class FakeResult:
    def __init__(self, counts_list: list[dict[str, int]], metadata: dict | None = None):
        self._counts_list = counts_list
        self._metadata = metadata or {}

    def get_counts(self):
        return self._counts_list


class FakeJob:
    def __init__(self, result: FakeResult):
        self._result = result

    def result(self):
        return self._result


def fake_run_binomial_last_bit(n_qubits: int, shots: int, rng: np.random.Generator, p_last_zero: float):
    """Return a ``backend.run``-like callable: one independent Binomial draw per circuit in the batch."""

    def run(circuits, shots_arg=None, **_kwargs):
        s = shots_arg if shots_arg is not None else shots
        counts_list = []
        for _ in circuits:
            n0 = int(rng.binomial(s, p_last_zero))
            counts_list.append(counts_for_last_bit(n_qubits, s, n0))
        return FakeJob(FakeResult(counts_list))

    return run
