"""Empirical variance of ``counts_to_expectation`` matches binomial shot noise."""

import numpy as np
import pytest

from odra_test.expectation_from_counts import counts_to_expectation
from odra_test.fake_shot_job import counts_for_last_bit


@pytest.mark.parametrize("n_qubits", [5])
@pytest.mark.parametrize("shots", [512, 1024])
@pytest.mark.parametrize("p", [0.35, 0.5, 0.72])
def test_expectation_variance_matches_binomial(n_qubits, shots, p):
    rng = np.random.default_rng(2026)
    n_trials = 4000
    evs = []
    for _ in range(n_trials):
        n0 = int(rng.binomial(shots, p))
        cts = counts_for_last_bit(n_qubits, shots, n0)
        evs.append(counts_to_expectation(cts))
    evs = np.array(evs)
    empirical = float(np.var(evs, ddof=1))
    theory = 4.0 * p * (1.0 - p) / shots
    ratio = empirical / theory if theory > 0 else 0.0
    assert 0.75 < ratio < 1.25, f"emp_var={empirical:.6g} theory={theory:.6g} ratio={ratio:.3f}"


def test_counts_to_expectation_extremes():
    c_all0 = {"00000": 100}
    assert counts_to_expectation(c_all0) == pytest.approx(1.0)
    c_all1 = {"00001": 100}
    assert counts_to_expectation(c_all1) == pytest.approx(-1.0)
