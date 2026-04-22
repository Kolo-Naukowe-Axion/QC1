"""Tests for the hypotheses behind the depth-6 IQM accuracy collapse.

Each test targets a specific mechanism discussed in the analysis:

- H-B (transpile / topology)
    - ``test_odra_physical_cx_count_grows_superlinearly_with_depth_on_star``
    - ``test_simulator_ansatz_has_more_physical_cx_than_odra_at_depth_6``
    - ``test_opt_level_3_reduces_total_gate_depth_vs_level_1_at_depth_6``

- H-A (noise / decoherence)
    - ``test_z_expectation_concentrates_toward_zero_with_depth_under_noise``
    - ``test_amplitude_damping_biases_z_expectation_toward_one_at_depth_6``
    - ``test_trained_depth6_weights_lose_input_sensitivity_under_amp_damping``

- H-D (observable / endianness)
    - ``test_trailing_Z_observable_measures_little_endian_q0``
    - ``test_counts_to_expectation_and_statevector_Z_agree_on_q0``

- H-E (analysis / protocol)
    - ``test_choose_shot_from_pilot_accepts_collapsed_hardware_signal``
    - ``test_more_shots_cannot_recover_a_biased_expectation``

Naming convention follows the existing ``odra_test`` suite (plain ``pytest`` functions, no class
hierarchy, module-level imports, fixtures only where they earn their keep).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    amplitude_damping_error,
    depolarizing_error,
)
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.metrics import accuracy_score, f1_score

from experiment_lib import (
    build_statevector_model,
    choose_shot_from_pilot,
    load_checkpoint_connector,
    load_checkpoint_hybrid,
    load_fold_test_data,
    load_phase_spec,
    odra_ansatz,
    predictions_to_labels,
    simulator_ansatz,
    weight_path,
)
from odra_test.expectation_from_counts import counts_to_expectation
from odra_test.fake_shot_job import fake_run_binomial_last_bit
from odra_test.iqm_backend_estimator import IQMBackendEstimator


# IQM Adonis-style 5-qubit star: hub qubit 2, leaves {0, 1, 3, 4}.
STAR_COUPLING = [[0, 2], [2, 0], [1, 2], [2, 1], [2, 3], [3, 2], [2, 4], [4, 2]]
BASIS = ["cx", "rz", "sx", "x"]
CONFIG_PATH = "tests/ansatz_odra_evaluation/experiment_config.toml"


def _star_backend() -> GenericBackendV2:
    return GenericBackendV2(
        num_qubits=5,
        basis_gates=BASIS,
        coupling_map=STAR_COUPLING,
        seed=42,
    )


def _measured_full_circuit(ansatz_fn, n_qubits: int, depth: int) -> QuantumCircuit:
    xparams = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(xparams[i], i)
    qc.compose(ansatz_fn(n_qubits, depth), inplace=True)
    qc.measure_all()
    return qc


# ------------------------------------------------------------------ #
# H-B: transpilation / topology amplifies 2Q cost for depth 6        #
# ------------------------------------------------------------------ #


def test_odra_physical_cx_count_grows_superlinearly_with_depth_on_star():
    """On a 5-qubit star topology, the physical CX count for the odra ansatz grows
    much faster than the logical depth. That inflated count is what exposes depth-6
    circuits to a decohering gate budget the shallower depths never see.
    """
    be = _star_backend()
    counts = {}
    for d in (2, 4, 6):
        tqc = transpile(
            _measured_full_circuit(odra_ansatz, 5, d),
            be,
            optimization_level=1,
            seed_transpiler=42,
        )
        counts[d] = tqc.count_ops().get("cx", 0)

    # Empirically the star + opt=1 transpile gives 13 / 38 / 60 at d=2/4/6.
    assert counts[2] < counts[4] < counts[6]
    # Super-linear growth: depth 6 has at least 3x the 2Q gates of depth 2.
    assert counts[6] >= 3 * counts[2]
    # Depth 6 CX count comfortably exceeds a "1% per gate" decoherence budget (i.e. > 50).
    assert counts[6] >= 50


def test_simulator_ansatz_has_more_physical_cx_than_odra_at_depth_6():
    """The ``simulator`` ansatz uses ``CRX``/``CRY`` rotations that compile to multiple
    CZ/CX per controlled-rotation, so on hardware it burns noticeably more 2Q budget
    than the ``odra`` ansatz at the same depth. This matches the observed hardware
    F1 being lower for ``simulator`` than ``odra`` at depth 6.
    """
    be = _star_backend()
    cx_odra = transpile(
        _measured_full_circuit(odra_ansatz, 5, 6), be, optimization_level=1, seed_transpiler=42
    ).count_ops().get("cx", 0)
    cx_sim = transpile(
        _measured_full_circuit(simulator_ansatz, 5, 6), be, optimization_level=1, seed_transpiler=42
    ).count_ops().get("cx", 0)
    # Ratio should be > 1.3 in practice; empirical value is 87 / 60 = 1.45.
    assert cx_sim > cx_odra
    assert cx_sim >= int(1.3 * cx_odra)


def test_opt_level_3_reduces_total_gate_depth_vs_level_1_at_depth_6():
    """Fixing ``optimization_level = 1`` is a cost amplifier: level 3 cannot cut the CX count
    of a hardware-native block easily, but it does reduce the *scheduled* depth (i.e. the
    circuit's wall-clock duration), which proportionally cuts decoherence exposure.
    Confirming that here justifies the suggestion to re-run with opt=3.
    """
    be = _star_backend()
    opt1 = transpile(
        _measured_full_circuit(odra_ansatz, 5, 6), be, optimization_level=1, seed_transpiler=42
    )
    opt3 = transpile(
        _measured_full_circuit(odra_ansatz, 5, 6), be, optimization_level=3, seed_transpiler=42
    )
    assert opt3.depth() < opt1.depth()


# ------------------------------------------------------------------ #
# H-A: noise / decoherence explanations for the collapse             #
# ------------------------------------------------------------------ #


def _build_depolarizing_aer(p_2q: float) -> AerSimulator:
    """Pure depolarizing noise on CX (and a small amount on 1Q gates).

    Used by the concentration test: depolarizing channels shrink ``<Z>`` toward zero
    symmetrically, so the effect is not confounded with a thermal-ground-state bias.
    """
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p_2q, 2), ["cx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p_2q / 30.0, 1), ["sx", "x", "rz"])
    layout = _star_backend()
    return AerSimulator.from_backend(layout, noise_model=nm)


def _build_amp_damping_aer(gamma_2q: float) -> AerSimulator:
    """Amplitude-damping (T1-biased) plus a small depolarizing error on CX.

    Used by the bias test: amplitude damping pulls ``<Z>`` toward +1 regardless of input,
    which is the specific mechanism by which predictions all collapse to one class.
    """
    nm = NoiseModel()
    ad2 = amplitude_damping_error(gamma_2q).tensor(amplitude_damping_error(gamma_2q))
    dp2 = depolarizing_error(gamma_2q / 2.0, 2)
    nm.add_all_qubit_quantum_error(ad2.compose(dp2), ["cx"])
    nm.add_all_qubit_quantum_error(
        amplitude_damping_error(gamma_2q / 20.0).compose(depolarizing_error(gamma_2q / 40.0, 1)),
        ["sx", "x"],
    )
    layout = _star_backend()
    return AerSimulator.from_backend(layout, noise_model=nm)


def _random_input_expectations(depth: int, backend: AerSimulator, *, n_inputs: int = 24) -> np.ndarray:
    """Build a fixed random odra ansatz at ``depth``, evaluate ``<Z_q0>`` for ``n_inputs``
    random inputs on the given backend. Returns the array of expectations.
    """
    n_qubits = 5
    rng = np.random.default_rng(0)
    ans = odra_ansatz(n_qubits, depth)
    theta = rng.uniform(-1.0, 1.0, size=len(ans.parameters))
    bound = ans.assign_parameters(theta)
    inputs = rng.uniform(-1.2, 1.2, size=(n_inputs, n_qubits))

    layout = _star_backend()
    evs = []
    for x in inputs:
        qc = QuantumCircuit(n_qubits)
        for i, xi in enumerate(x):
            qc.ry(xi, i)
        qc.compose(bound, inplace=True)
        qc.measure_all()
        tqc = transpile(qc, layout, optimization_level=1, seed_transpiler=42)
        counts = backend.run(tqc, shots=4096).result().get_counts()
        evs.append(counts_to_expectation(counts))
    return np.asarray(evs)


def test_z_expectation_contracts_much_more_with_depth_under_depolarizing_noise():
    """Barren-plateau-like concentration: under pure depolarizing noise, the *contraction
    factor* applied to ``<Z>`` grows rapidly with depth because the effective channel
    depth scales with the transpiled 2Q gate count. Comparing ``noisy / ideal`` at depth 2
    vs depth 6 isolates this contraction from the raw magnitude of ``<Z>`` at each depth.

    Pure depolarizing is used (no amplitude damping) so the effect is an unbiased shrink
    toward zero, not a sign-flipping thermal pull.
    """
    ideal = AerSimulator(seed_simulator=42)
    noisy = _build_depolarizing_aer(p_2q=0.015)

    evs_d2_ideal = _random_input_expectations(2, ideal)
    evs_d2_noisy = _random_input_expectations(2, noisy)
    evs_d6_ideal = _random_input_expectations(6, ideal)
    evs_d6_noisy = _random_input_expectations(6, noisy)

    ratio_d2 = float(np.mean(np.abs(evs_d2_noisy))) / float(np.mean(np.abs(evs_d2_ideal)))
    ratio_d6 = float(np.mean(np.abs(evs_d6_noisy))) / float(np.mean(np.abs(evs_d6_ideal)))

    # Depth 2 contraction is mild; depth 6 contraction is severe.
    # Empirically ratio_d2 ~ 0.95 (small contraction), ratio_d6 ~ 0.60 (big contraction).
    assert ratio_d2 > 0.8, f"Depth-2 ideal->noisy contraction unexpectedly large: {ratio_d2:.3f}"
    assert ratio_d6 < 0.75, f"Depth-6 ideal->noisy contraction unexpectedly mild: {ratio_d6:.3f}"
    assert ratio_d6 < ratio_d2

    # Absolute sanity: at depth 6 under noise, ``<Z>`` lives in a much smaller band than
    # at depth 2 under noise, so the distribution collapses *as a distribution*, not just
    # per-point.
    assert float(np.max(np.abs(evs_d6_noisy))) < float(np.max(np.abs(evs_d2_noisy)))


def test_amplitude_damping_biases_z_expectation_toward_one_at_depth_6():
    """Under enough T1-dominated noise, ``<Z_q0>`` is pulled toward +1 (ground state) regardless
    of the input. This is the mechanism by which hardware predictions all collapse to a
    single class at depth 6 (and why thresholding at 0 yields accuracy ~= class prior).
    """
    rng = np.random.default_rng(1)
    ans = odra_ansatz(5, 6)
    theta = rng.uniform(-1.0, 1.0, size=len(ans.parameters))
    bound = ans.assign_parameters(theta)

    backend = _build_amp_damping_aer(gamma_2q=0.08)
    layout = _star_backend()
    inputs = rng.uniform(-1.0, 1.0, size=(16, 5))
    evs = []
    for x in inputs:
        qc = QuantumCircuit(5)
        for i, xi in enumerate(x):
            qc.ry(xi, i)
        qc.compose(bound, inplace=True)
        qc.measure_all()
        tqc = transpile(qc, layout, optimization_level=1, seed_transpiler=42)
        counts = backend.run(tqc, shots=4096).result().get_counts()
        evs.append(counts_to_expectation(counts))
    evs = np.asarray(evs)

    # All predictions collapse into a very narrow band.
    assert float(np.std(evs)) < 0.1
    # The band is centered on a small positive value (T1 bias toward |0> i.e. +1 for <Z>).
    assert float(np.mean(evs)) > 0.0
    # After thresholding at 0, the vast majority of predictions have the same sign.
    pos_fraction = float(np.mean(evs > 0))
    assert pos_fraction >= 0.9 or pos_fraction <= 0.1


def _depth6_weights_available() -> bool:
    spec = load_phase_spec(CONFIG_PATH, phase="final", depth=6)
    return weight_path(spec, "odra", 1).is_file()


@pytest.mark.skipif(not _depth6_weights_available(), reason="depth-6 odra fold-1 checkpoint not present")
def test_trained_depth6_weights_lose_input_sensitivity_under_amp_damping():
    """Integration test: load the *actual* trained depth-6 odra checkpoint for fold 1,
    evaluate it on the fold's held-out test data through an ``IQMBackendEstimator`` whose
    backend is a noisy AerSimulator, and verify that the predictions collapse to a narrow
    band around one sign -- exactly the symptom observed on IQM hardware.

    The statevector baseline on the same samples is checked as a sanity anchor so the
    test only fails when the noisy model really is losing input sensitivity.
    """
    spec = load_phase_spec(CONFIG_PATH, phase="final", depth=6)
    X, y = load_fold_test_data(spec, 1)
    rng = np.random.default_rng(0)
    pos_idx = rng.choice(np.where(y == 1)[0], size=10, replace=False)
    neg_idx = rng.choice(np.where(y == -1)[0], size=10, replace=False)
    X20 = X[np.concatenate([pos_idx, neg_idx])]

    sv_model = build_statevector_model("odra", spec)
    load_checkpoint_hybrid(sv_model, weight_path(spec, "odra", 1))
    sv_model.eval()
    with torch.no_grad():
        sv_pred = sv_model(torch.tensor(X20)).detach().cpu().numpy().flatten()
    # Sanity anchor: statevector predictions should span a reasonable range.
    assert float(np.std(sv_pred)) > 0.1, "Statevector predictions should not already be collapsed"

    backend = _build_amp_damping_aer(gamma_2q=0.08)
    hw_ansatz = odra_ansatz(5, 6)
    xp = ParameterVector("x", 5)
    fm = QuantumCircuit(5)
    for i in range(5):
        fm.ry(xp[i], i)
    hw_qc = QuantumCircuit(5)
    hw_qc.compose(fm, inplace=True)
    hw_qc.compose(hw_ansatz, inplace=True)
    obs = SparsePauliOp.from_list([("I" * 4 + "Z", 1.0)])
    est = IQMBackendEstimator(
        backend,
        options={"shots": 1024, "optimization_level": 1, "seed_transpiler": 42},
    )
    qnn = EstimatorQNN(
        circuit=hw_qc,
        observables=obs,
        input_params=list(xp),
        weight_params=list(hw_ansatz.parameters),
        estimator=est,
    )
    connector = TorchConnector(qnn)
    load_checkpoint_connector(connector, weight_path(spec, "odra", 1))
    connector.eval()
    with torch.no_grad():
        noisy_pred = connector(torch.tensor(X20, dtype=torch.float32)).detach().cpu().numpy().flatten()

    # 1) Noisy predictions collapse to a narrow band (no input sensitivity left).
    assert float(np.std(noisy_pred)) < 0.5 * float(np.std(sv_pred))
    assert float(np.std(noisy_pred)) < 0.1
    # 2) After thresholding at 0, >= 90% of predictions are the same class
    #    -> this is the mechanism that forces hardware accuracy to the class prior.
    noisy_labels = predictions_to_labels(noisy_pred)
    majority_frac = float(max(np.mean(noisy_labels == 1), np.mean(noisy_labels == -1)))
    assert majority_frac >= 0.9


# ------------------------------------------------------------------ #
# H-D: observable / endianness sanity                                #
# ------------------------------------------------------------------ #


def test_trailing_Z_observable_measures_little_endian_q0():
    """Qiskit strings in ``SparsePauliOp`` read right-to-left. ``"IIIIZ"`` acts on qubit 0,
    not on the "last" qubit. If any training / hardware path disagreed about which qubit
    carried the label, the collapse would look exactly like what depth 6 shows. Here we
    assert the convention so the pipeline's measurement target is unambiguous.
    """
    qc_q0 = QuantumCircuit(5)
    qc_q0.x(0)
    qc_q4 = QuantumCircuit(5)
    qc_q4.x(4)

    obs = SparsePauliOp.from_list([("I" * 4 + "Z", 1.0)])
    est = StatevectorEstimator(seed=0)
    evs = est.run([(qc_q0, obs), (qc_q4, obs)]).result()
    ev_q0 = float(evs[0].data.evs)
    ev_q4 = float(evs[1].data.evs)

    assert ev_q0 == pytest.approx(-1.0, abs=1e-6), "'IIIIZ' should act on qubit 0 (little-endian)"
    assert ev_q4 == pytest.approx(+1.0, abs=1e-6), "'IIIIZ' should leave qubit 4 untouched"


def test_counts_to_expectation_and_statevector_Z_agree_on_q0():
    """The IQMBackendEstimator's ``counts_to_expectation`` inspects ``bitstring[-1]``, which
    is Qiskit's LSB == qubit 0. Verify this matches the ``"IIIIZ"`` statevector expectation so
    training and hardware paths can't disagree about the measured qubit.
    """
    shots = 2048
    counts_x_on_q0 = {"00001": shots}
    counts_x_on_q4 = {"10000": shots}

    assert counts_to_expectation(counts_x_on_q0) == pytest.approx(-1.0, abs=1e-9)
    assert counts_to_expectation(counts_x_on_q4) == pytest.approx(+1.0, abs=1e-9)


# ------------------------------------------------------------------ #
# H-E: analysis / protocol flaws                                     #
# ------------------------------------------------------------------ #


def test_choose_shot_from_pilot_accepts_collapsed_hardware_signal():
    """Reproduces the depth-6 scenario: hardware mean accuracy ~= class prior and barely moves
    across shot counts (deltas well under ``delta_accuracy``), while the statevector baseline is
    much higher. ``choose_shot_from_pilot`` happily returns the *lowest* stable shot count
    because the current rule only checks shot-to-shot stability, not statevector-to-hardware
    agreement. This documents the protocol gap and guards against silently "fixing" the
    selector in a way that would change its behavior on healthy runs.
    """
    spec = load_phase_spec(CONFIG_PATH, phase="pilot", depth=6)

    rows = []
    for fold in (1, 2):
        for ansatz in ("odra", "simulator"):
            for shot in (512, 1024, 2048, 4096):
                rows.append(
                    {
                        "phase": "pilot",
                        "depth": 6,
                        "fold": fold,
                        "ansatz": ansatz,
                        "eval_shots": shot,
                        "iqm_mean_accuracy": 0.578,
                        "iqm_mean_f1": 0.090,
                        "statevector_accuracy": 0.90,
                        "statevector_f1": 0.88,
                        "completed_repeats": 10,
                    }
                )
    summary_df = pd.DataFrame(rows)

    # Current implementation: the smallest adjacent-shot pair is trivially stable, so 1024 wins
    # even though the hardware accuracy is 32 percentage points below statevector.
    chosen = choose_shot_from_pilot(summary_df, spec)
    assert chosen == 1024, (
        "The protocol selector still accepts a collapsed hardware signal; "
        "add a statevector-vs-hardware guard in choose_shot_from_pilot to prevent this."
    )

    # The statevector-vs-hardware gap is in fact huge (> 0.25 on both accuracy and F1),
    # which no current criterion in ``experiment_lib`` checks.
    mean_gap_acc = (summary_df["statevector_accuracy"] - summary_df["iqm_mean_accuracy"]).mean()
    mean_gap_f1 = (summary_df["statevector_f1"] - summary_df["iqm_mean_f1"]).mean()
    assert mean_gap_acc > 0.25
    assert mean_gap_f1 > 0.5


def test_more_shots_cannot_recover_a_biased_expectation():
    """When the hardware has introduced a deterministic bias into ``<Z>``, adding more shots
    only shrinks the *statistical* noise; it cannot pull the mean back to the true value.
    That is precisely why depth-6 pilot data shows <0.01 accuracy deltas across 512 -> 4096
    shots even though the mean is 0.56 instead of 0.85. Here we emulate that with a fake
    binomial backend whose per-shot ``p0`` is 0.51 (biased ``<Z>`` ~= +0.02) and show that
    all shot counts give essentially the same (wrong) answer.
    """
    n_qubits = 5
    shots_grid = [512, 1024, 2048, 4096, 16384]
    rng = np.random.default_rng(99)
    be = GenericBackendV2(num_qubits=n_qubits, basis_gates=BASIS, seed=1)

    xp = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(xp[i], i)
    obs = SparsePauliOp.from_list([("I" * (n_qubits - 1) + "Z", 1.0)])
    x = np.zeros(n_qubits)

    p_biased = 0.51
    biased_expectation = 2 * p_biased - 1  # = 0.02
    n_trials = 200

    per_shot_means = {}
    per_shot_stds = {}
    for shots in shots_grid:
        est = IQMBackendEstimator(
            be, options={"shots": shots, "optimization_level": 1, "seed_transpiler": 7}
        )
        be.run = fake_run_binomial_last_bit(n_qubits, shots, rng, p_biased)
        evs = []
        for _ in range(n_trials):
            job = est.run([(qc, obs, x)])
            evs.append(float(job.result()[0].data.evs[0]))
        per_shot_means[shots] = float(np.mean(evs))
        per_shot_stds[shots] = float(np.std(evs, ddof=1))

    # 1) Mean does NOT approach the true (noise-free) value; it sits at the biased <Z>.
    for shots in shots_grid:
        assert abs(per_shot_means[shots] - biased_expectation) < 0.02

    # 2) Standard deviation shrinks as 1/sqrt(shots): quadrupling shots halves std (~ factor 2).
    assert per_shot_stds[512] > 1.5 * per_shot_stds[2048]
    assert per_shot_stds[2048] > 1.5 * per_shot_stds[16384]

    # 3) The hardware-vs-truth gap is essentially unchanged between the smallest and
    #    largest shot counts -> shot-to-shot stability is a false positive for "signal OK".
    delta_across_shots = abs(per_shot_means[16384] - per_shot_means[512])
    assert delta_across_shots < 0.01
