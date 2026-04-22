## Ansatz ODRA Evaluation

This directory now has a script-first experiment layout for the IQM Spark ansatz comparison.

### What changed

- Shared experiment logic lives in `experiment_lib.py` instead of being duplicated across notebooks.
- Runtime configuration lives in `experiment_config.toml`.
- Pilot selection, final execution, and final analysis have dedicated entrypoints.
- Notebook outputs can be treated as exploratory artifacts rather than the primary execution path.

### Main entrypoints

- `run_cv_experiment.py`
  - Runs one experiment phase (`pilot` or `final`) for one depth.
  - Produces checkpoint-safe CSV outputs.
- `select_protocol.py`
  - Reads a pilot run and recommends a frozen shot count and repeat count.
- `analyze_final_experiment.py`
  - Produces fold-level summaries, paired differences, and exact Wilcoxon/sign-test outputs.

### Config

The protocol lives in `experiment_config.toml`.

- `pilot` uses a shot grid and smaller fold subset.
- `final` uses a single chosen shot count and full-fold evaluation.
- Depth-specific checkpoint rules are encoded once in the config for depths `2`, `4`, and `6`.
- The experiment config intentionally fixes `optimization_level = 1`, `seed_transpiler = 42`, and `checkpoint_epoch = 30` across depths to keep routing and compilation behavior reproducible and to avoid aggressive transpilation adapting to the current IQM Spark calibration state.

### Outputs

Each run writes into:

- `outputs/<phase>/<run_id>/statevector_results.csv`
- `outputs/<phase>/<run_id>/run_level_results.csv`
- `outputs/<phase>/<run_id>/summary_comparison.csv`
- `outputs/<phase>/<run_id>/run_manifest.json`

Analysis scripts add:

- `outputs/<phase>/<run_id>/ansatz_level_summary.csv`
- `outputs/<phase>/<run_id>/paired_fold_differences.csv`
- `outputs/<phase>/<run_id>/paired_tests.csv`
- `outputs/<phase>/<run_id>/shot_stability.csv` for pilot runs

### Typical workflow

Run the same workflow independently for each depth in `{2, 4, 6}`.

Pilot for a chosen depth:

```bash
DEPTH=2
uv run python tests/ansatz_odra_evaluation/run_cv_experiment.py --phase pilot --depth "${DEPTH}" --run-id "pilot_depth${DEPTH}"
uv run python tests/ansatz_odra_evaluation/select_protocol.py --run-dir "tests/ansatz_odra_evaluation/outputs/pilot/pilot_depth${DEPTH}"
```

Final run for the same depth after freezing the protocol:

```bash
DEPTH=2
uv run python tests/ansatz_odra_evaluation/run_cv_experiment.py --phase final --depth "${DEPTH}" --run-id "final_depth${DEPTH}"
uv run python tests/ansatz_odra_evaluation/analyze_final_experiment.py --run-dir "tests/ansatz_odra_evaluation/outputs/final/final_depth${DEPTH}"
```

To evaluate all depths, repeat that sequence for `DEPTH=2`, then `4`, then `6`.

### Notes

- Scripts use `IQM_TOKEN` from the environment when available, and only prompt if it is missing.
- Hardware execution order is interleaved by repeat to reduce time-drift bias between ansatze.
- Existing notebooks remain available, but the scripts are now the recommended path for reproducible runs.
