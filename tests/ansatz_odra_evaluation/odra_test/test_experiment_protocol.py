from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from experiment_lib import (
    append_csv_row,
    choose_shot_from_pilot,
    completed_task_keys,
    load_phase_spec,
    summarize_results,
    read_csv_or_empty,
    recommended_repeats_from_pilot,
    successful_run_df,
    wilcoxon_signed_rank_exact,
)


def test_wilcoxon_signed_rank_exact_all_positive_differences():
    result = wilcoxon_signed_rank_exact([0.1, 0.2, 0.3, 0.4, 0.5])
    assert result["n_nonzero"] == 5
    assert math.isclose(result["statistic"], 0.0)
    assert math.isclose(result["pvalue"], 0.0625)
    assert math.isclose(result["rank_biserial"], 1.0)


def test_choose_shot_from_pilot_uses_first_stable_adjacent_level():
    spec = load_phase_spec(
        "tests/ansatz_odra_evaluation/experiment_config.toml",
        phase="pilot",
        depth=4,
    )
    summary_df = pd.DataFrame(
        [
            {"phase": "pilot", "depth": 4, "fold": 1, "ansatz": "odra", "eval_shots": 512, "iqm_mean_accuracy": 0.80, "iqm_mean_f1": 0.78, "completed_repeats": 10},
            {"phase": "pilot", "depth": 4, "fold": 1, "ansatz": "odra", "eval_shots": 1024, "iqm_mean_accuracy": 0.82, "iqm_mean_f1": 0.80, "completed_repeats": 10},
            {"phase": "pilot", "depth": 4, "fold": 1, "ansatz": "odra", "eval_shots": 2048, "iqm_mean_accuracy": 0.828, "iqm_mean_f1": 0.809, "completed_repeats": 10},
            {"phase": "pilot", "depth": 4, "fold": 1, "ansatz": "odra", "eval_shots": 4096, "iqm_mean_accuracy": 0.829, "iqm_mean_f1": 0.810, "completed_repeats": 10},
            {"phase": "pilot", "depth": 4, "fold": 1, "ansatz": "simulator", "eval_shots": 512, "iqm_mean_accuracy": 0.74, "iqm_mean_f1": 0.68, "completed_repeats": 10},
            {"phase": "pilot", "depth": 4, "fold": 1, "ansatz": "simulator", "eval_shots": 1024, "iqm_mean_accuracy": 0.76, "iqm_mean_f1": 0.70, "completed_repeats": 10},
            {"phase": "pilot", "depth": 4, "fold": 1, "ansatz": "simulator", "eval_shots": 2048, "iqm_mean_accuracy": 0.768, "iqm_mean_f1": 0.714, "completed_repeats": 10},
            {"phase": "pilot", "depth": 4, "fold": 1, "ansatz": "simulator", "eval_shots": 4096, "iqm_mean_accuracy": 0.769, "iqm_mean_f1": 0.715, "completed_repeats": 10},
        ]
    )
    assert choose_shot_from_pilot(summary_df, spec) == 2048


def test_recommended_repeats_from_pilot_uses_most_variable_row():
    spec = load_phase_spec(
        "tests/ansatz_odra_evaluation/experiment_config.toml",
        phase="pilot",
        depth=4,
    )
    summary_df = pd.DataFrame(
        [
            {"eval_shots": 2048, "completed_repeats": 10, "iqm_std_accuracy": 0.02, "iqm_std_f1": 0.03},
            {"eval_shots": 2048, "completed_repeats": 10, "iqm_std_accuracy": 0.05, "iqm_std_f1": 0.08},
        ]
    )
    result = recommended_repeats_from_pilot(summary_df, 2048, spec)
    assert result["recommended_repeats_accuracy"] == math.ceil(((1.96 * 0.05) / 0.02) ** 2)
    assert result["recommended_repeats_f1"] == math.ceil(((1.96 * 0.08) / 0.03) ** 2)


def test_completed_task_keys_ignores_failed_legacy_rows():
    frame = pd.DataFrame(
        [
            {"fold": 1, "ansatz": "odra", "shots": 512, "repeat_index": 0, "qpu_time_total": 0.0},
            {"fold": 1, "ansatz": "simulator", "shots": 512, "repeat_index": 0, "qpu_time_total": 12.5},
        ]
    )
    assert completed_task_keys(frame) == {(1, "simulator", 512, 0)}


def test_completed_task_keys_prefers_explicit_success_status():
    frame = pd.DataFrame(
        [
            {"fold": 1, "ansatz": "odra", "shots": 512, "repeat_index": 0, "status": "failed", "qpu_time_total": 15.0},
            {"fold": 1, "ansatz": "simulator", "shots": 512, "repeat_index": 0, "status": "success", "qpu_time_total": 0.0},
        ]
    )
    assert completed_task_keys(frame) == {(1, "simulator", 512, 0)}


def test_read_csv_or_empty_normalizes_mixed_legacy_and_status_rows(tmp_path: Path):
    path = tmp_path / "run_level_results.csv"
    path.write_text(
        "\n".join(
            [
                "timestamp_utc,phase,depth,fold,ansatz,shots,repeat_index,accuracy,f1,weight_path,test_csv,n_samples,qpu_time_total,wall_time_forward_s,calibration_set_id,optimization_level,seed_transpiler",
                "2026-01-01T00:00:00+00:00,pilot,2,1,simulator,512,0,0.55,0.0,w,t,275,0.0,42.0,,1,42",
                "2026-01-01T00:01:00+00:00,success,pilot,2,1,simulator,512,0,0.83,0.77,w,t,275,57.0,70.0,,1,42",
            ]
        )
        + "\n"
    )
    frame = read_csv_or_empty(path)
    assert list(frame.columns)[0:3] == ["timestamp_utc", "status", "phase"]
    assert frame.loc[0, "status"] == "failed"
    assert frame.loc[1, "status"] == "success"


def test_append_csv_row_rewrites_mixed_schema_file(tmp_path: Path):
    path = tmp_path / "run_level_results.csv"
    path.write_text(
        "\n".join(
            [
                "timestamp_utc,phase,depth,fold,ansatz,shots,repeat_index,accuracy,f1,weight_path,test_csv,n_samples,qpu_time_total,wall_time_forward_s,calibration_set_id,optimization_level,seed_transpiler",
                "2026-01-01T00:00:00+00:00,pilot,2,1,simulator,512,0,0.55,0.0,w,t,275,0.0,42.0,,1,42",
            ]
        )
        + "\n"
    )
    append_csv_row(
        path,
        {
            "timestamp_utc": "2026-01-01T00:01:00+00:00",
            "status": "success",
            "phase": "pilot",
            "depth": 2,
            "fold": 1,
            "ansatz": "simulator",
            "shots": 512,
            "repeat_index": 0,
            "accuracy": 0.83,
            "f1": 0.77,
            "weight_path": "w",
            "test_csv": "t",
            "n_samples": 275,
            "qpu_time_total": 57.0,
            "wall_time_forward_s": 70.0,
            "calibration_set_id": "",
            "optimization_level": 1,
            "seed_transpiler": 42,
        },
    )
    frame = read_csv_or_empty(path)
    assert len(frame) == 2
    assert "status" in frame.columns
    assert successful_run_df(frame).shape[0] == 1


def test_summarize_results_ignores_failed_rows():
    spec = load_phase_spec(
        "tests/ansatz_odra_evaluation/experiment_config.toml",
        phase="pilot",
        depth=2,
    )
    statevector_df = pd.DataFrame(
        [
            {
                "phase": "pilot",
                "depth": 2,
                "fold": 1,
                "ansatz": "odra",
                "statevector_accuracy": 0.9,
                "statevector_f1": 0.8,
                "test_csv": "t",
                "weight_path": "w",
            }
        ]
    )
    run_df = pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-01-01T00:00:00+00:00",
                "status": "failed",
                "phase": "pilot",
                "depth": 2,
                "fold": 1,
                "ansatz": "odra",
                "shots": 512,
                "repeat_index": 0,
                "accuracy": 0.1,
                "f1": 0.0,
                "weight_path": "w",
                "test_csv": "t",
                "n_samples": 10,
                "qpu_time_total": 0.0,
                "wall_time_forward_s": 1.0,
                "calibration_set_id": "",
                "optimization_level": 1,
                "seed_transpiler": 42,
            },
            {
                "timestamp_utc": "2026-01-01T00:01:00+00:00",
                "status": "success",
                "phase": "pilot",
                "depth": 2,
                "fold": 1,
                "ansatz": "odra",
                "shots": 512,
                "repeat_index": 0,
                "accuracy": 0.85,
                "f1": 0.75,
                "weight_path": "w",
                "test_csv": "t",
                "n_samples": 10,
                "qpu_time_total": 12.0,
                "wall_time_forward_s": 13.0,
                "calibration_set_id": "",
                "optimization_level": 1,
                "seed_transpiler": 42,
            },
        ]
    )
    summary = summarize_results(spec, statevector_df=statevector_df, run_df=run_df)
    row = summary.iloc[0]
    assert row["iqm_mean_accuracy"] == 0.85
    assert row["iqm_mean_f1"] == 0.75
    assert row["completed_repeats"] == 1
