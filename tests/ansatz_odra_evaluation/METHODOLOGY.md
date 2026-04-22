# Ansatz ODRA Evaluation Methodology

## Objective

This experiment compares two pretrained 5-qubit variational ansatz families, `odra` and `simulator`, under the same cross-validation splits and evaluation protocol. The goal is to measure how closely each ansatz preserves its statevector performance when executed on IQM Odra hardware, and to compare the two ansatze in terms of classification quality and hardware robustness.

## Experimental Units

- Depths evaluated: `2`, `4`, and `6`
- Qubits: `5`
- Random seed: `42`
- Transpiler optimization level: `1`
- Transpiler seed: `42`
- Hardware backend URL: `https://odra5.e-science.pl/`

The experiment is evaluation-only. Model weights are not trained inside this workflow. Instead, each run loads fixed checkpoints from the cross-validation setup and evaluates them on held-out test folds.

## Data And Cross-Validation Setup

The evaluation uses predefined cross-validation folds stored under `setup/cross_validation/Data`. For each fold:

- `test_data.csv` contains 5 input features and 1 target column
- the first 5 columns are mapped to the 5 qubits
- the label column is converted to the binary set `{-1, +1}` if it is stored as `{0, 1}`

For every depth and fold, the script loads the matching pretrained checkpoint for each ansatz from the corresponding weights directory. The checkpoint epoch is fixed to `30` for all tested depths.

## Quantum Model Definition

Each evaluated model is a hybrid quantum classifier with the following fixed structure:

1. Classical inputs are angle-encoded with one `RY` rotation per qubit.
2. The encoded state is passed through either the `odra` ansatz or the `simulator` ansatz.
3. The model output is the expectation value of the Pauli `Z` observable on the last qubit.
4. Continuous predictions are converted to class labels with threshold `0`, yielding `+1` for positive values and `-1` otherwise.

The same checkpointed weights are used for:

- a statevector reference evaluation
- repeated hardware executions on IQM Odra

This isolates hardware-induced performance changes from training differences.

## Ansatz Comparison Principle

The comparison is controlled so that both ansatze are evaluated:

- on the same folds
- with the same test samples
- at the same circuit depth
- with the same shot budget within a given run
- using the same transpilation settings

This means the main experimental variable is the ansatz choice, while data split, evaluation procedure, and hardware protocol are held fixed.

## Phase Structure

The workflow is split into two phases: `pilot` and `final`.

### Pilot Phase

The pilot phase is used to select a stable hardware protocol before the full evaluation.

- Folds: `1`, `2`
- Shots tested: `512`, `1024`, `2048`, `4096`
- Repeats per `(fold, ansatz, shots)` condition: `10`
- Hardware execution enabled: yes

For each depth, the pilot produces repeated hardware estimates and checks how much the mean hardware metrics change between consecutive shot counts.

The shot-selection rule chooses the first shot count whose worst-case change satisfies:

- maximum absolute change in accuracy `<= 0.01`
- maximum absolute change in F1 `<= 0.02`

If no shot count satisfies those stability thresholds, the protocol falls back to the largest tested shot count.

After choosing the pilot shot count, the script recommends the number of hardware repeats needed to achieve target 95% confidence-interval half-widths of:

- accuracy half-width `0.02`
- F1 half-width `0.03`

The repeat recommendation is based on the most conservative observed hardware standard deviation at the chosen shot count.

### Final Phase

The final phase runs the frozen protocol on the full cross-validation set.

- Folds: `1`, `2`, `3`, `4`, `5`
- Shots: `2048`
- Repeats per `(fold, ansatz)` condition: `20`
- Hardware execution enabled: yes

In the current configuration, the final protocol is already frozen in `experiment_config.toml` as `2048` shots and `20` repeats.

## Execution Procedure

For each selected depth, the execution proceeds as follows:

1. Compute one statevector reference result for every `(fold, ansatz)` pair.
2. Enumerate hardware tasks over all combinations of fold, shot count, repeat index, and ansatz.
3. For each hardware task, load the pretrained checkpoint, run inference on the full test fold, and record metrics.
4. Update summary files incrementally so interrupted runs can be resumed without recomputing completed tasks.

To reduce drift-related bias, the ansatz order is shuffled within each `(fold, shots, repeat)` block using the fixed random seed. This prevents one ansatz from always running earlier or later in the hardware queue.

The execution script also retries transient hardware failures with exponential backoff. Failed tasks are not silently written as completed results, so reruns preserve data integrity.

## Measured Outcomes

The primary performance metrics are:

- classification accuracy
- F1 score for the positive class

For the statevector baseline, one value per `(fold, ansatz)` is recorded.

For hardware execution, repeated runs are summarized per `(fold, ansatz, shots)` by:

- mean accuracy
- standard deviation of accuracy
- mean F1
- standard deviation of F1
- number of completed repeats

The experiment also records execution metadata such as total QPU time, wall-clock forward-pass time, calibration-set identifier when available, and the exact checkpoint and test file used.

## Comparative Analysis

The final analysis aggregates results in three complementary ways:

### 1. Across-fold ansatz summaries

For each `(phase, depth, ansatz, shots)` condition, the analysis computes:

- mean statevector accuracy and F1 across folds
- mean hardware accuracy and F1 across folds
- variability of hardware means across folds
- mean statevector-to-hardware performance gap

### 2. Paired fold differences

Within each fold, the two ansatze are compared directly by computing:

- hardware accuracy difference: `odra - simulator`
- hardware F1 difference: `odra - simulator`
- difference in statevector-to-hardware accuracy gap
- difference in statevector-to-hardware F1 gap
- difference in hardware standard deviations

This paired design controls for fold-specific difficulty because both ansatze are evaluated on the same held-out data.

### 3. Exact paired significance tests

For each depth and shot setting, the analysis applies:

- exact Wilcoxon signed-rank tests
- exact sign tests

These tests are run on fold-level paired differences rather than on pooled sample-level predictions, which keeps the unit of inference aligned with the cross-validation design.

## Reproducibility Controls

The following settings are fixed to improve reproducibility:

- predefined train/test folds
- fixed checkpoint epoch (`30`)
- fixed random seed (`42`)
- fixed transpiler seed (`42`)
- fixed transpiler optimization level (`1`)
- script-based execution and manifest logging

Each run writes a manifest plus CSV outputs so the evaluation can be resumed, audited, and reanalyzed without rerunning completed tasks.

## Output Artifacts

Each run writes the following core files:

- `run_manifest.json`
- `statevector_results.csv`
- `run_level_results.csv`
- `summary_comparison.csv`

Analysis steps may additionally create:

- `ansatz_level_summary.csv`
- `paired_fold_differences.csv`
- `paired_tests.csv`
- `shot_stability.csv`
- `shot_stability_aggregate.csv`
- `protocol_recommendation.json`

## Interpretation

The central methodological question is not only which ansatz has higher raw accuracy or F1, but also which ansatz transfers more reliably from ideal statevector evaluation to repeated real-hardware execution on IQM Odra. The pilot phase is therefore used to freeze a hardware protocol that is stable enough for comparison, and the final phase uses paired fold-level analyses to compare both ansatze under matched conditions.
