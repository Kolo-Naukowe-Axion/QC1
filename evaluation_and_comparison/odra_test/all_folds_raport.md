# All-Fold Evaluation Report

## Scope

This report summarizes the executed results from `evaluation_and_comparison/all_folds_evaluation.ipynb` together with the saved console output in `/Users/jkw/Downloads/output.txt`.

The run used:

- folds: `1-5`
- model sources: `ideal` and `noise`
- checkpoint: `final`
- IQM shot count: `512`

Each fold was evaluated on its own test split with:

- `StatevectorEstimator`
- IQM hardware backend

## Main Results

### Per-fold results


| Fold | Source | Backend     | Accuracy | F1     | Time / sample [s] |
| ---- | ------ | ----------- | -------- | ------ | ----------------- |
| 1    | ideal  | Statevector | 0.8364   | 0.7982 | 0.000718          |
| 1    | ideal  | IQM 512     | 0.8400   | 0.8346 | 0.207469          |
| 1    | noise  | Statevector | 0.8109   | 0.7570 | 0.000659          |
| 1    | noise  | IQM 512     | 0.7964   | 0.8042 | 0.212365          |
| 2    | ideal  | Statevector | 0.9055   | 0.8898 | 0.000680          |
| 2    | ideal  | IQM 512     | 0.8655   | 0.8593 | 0.207476          |
| 2    | noise  | Statevector | 0.9091   | 0.8927 | 0.000755          |
| 2    | noise  | IQM 512     | 0.8545   | 0.8462 | 0.207435          |
| 3    | ideal  | Statevector | 0.8540   | 0.8077 | 0.000748          |
| 3    | ideal  | IQM 512     | 0.8686   | 0.8364 | 0.207552          |
| 3    | noise  | Statevector | 0.8577   | 0.8134 | 0.000724          |
| 3    | noise  | IQM 512     | 0.7883   | 0.7803 | 0.207489          |
| 4    | ideal  | Statevector | 0.8832   | 0.8571 | 0.000723          |
| 4    | ideal  | IQM 512     | 0.8467   | 0.8467 | 0.207498          |
| 4    | noise  | Statevector | 0.8905   | 0.8649 | 0.000716          |
| 4    | noise  | IQM 512     | 0.8650   | 0.8384 | 0.207544          |
| 5    | ideal  | Statevector | 0.8832   | 0.8730 | 0.000721          |
| 5    | ideal  | IQM 512     | 0.8467   | 0.8511 | 0.207461          |
| 5    | noise  | Statevector | 0.8723   | 0.8583 | 0.000655          |
| 5    | noise  | IQM 512     | 0.8650   | 0.8582 | 0.207505          |


### Aggregate results across folds


| Source | Backend     | Mean accuracy | Std accuracy | Mean F1 | Std F1 | Mean time / sample [s] |
| ------ | ----------- | ------------- | ------------ | ------- | ------ | ---------------------- |
| ideal  | Statevector | 0.8725        | 0.0272       | 0.8452  | 0.0404 | 0.000718               |
| ideal  | IQM 512     | 0.8535        | 0.0127       | 0.8456  | 0.0103 | 0.207491               |
| noise  | Statevector | 0.8681        | 0.0374       | 0.8373  | 0.0531 | 0.000702               |
| noise  | IQM 512     | 0.8338        | 0.0382       | 0.8255  | 0.0323 | 0.208468               |


Note: the saved `output.txt` cuts off the last aggregate row. The `noise + Statevector` aggregate above was reconstructed from the full detailed fold table.

## Interpretation

### 1. Best overall average performance

The best average accuracy was achieved by the `ideal` model on `StatevectorEstimator`:

- mean accuracy: `0.8725`
- mean F1: `0.8452`

The `noise` statevector model was close, but slightly worse on average:

- accuracy lower by about `0.0044`
- F1 lower by about `0.0079`

### 2. Hardware penalty

Moving from `StatevectorEstimator` to IQM at `512` shots reduced average accuracy for both training variants:

- `ideal`: `0.8725 -> 0.8535` (`-0.0190`)
- `noise`: `0.8681 -> 0.8338` (`-0.0343`)

For F1:

- `ideal`: `0.8452 -> 0.8456` (effectively unchanged)
- `noise`: `0.8373 -> 0.8255` (`-0.0118`)

This suggests that the ideal-trained model transfers to hardware better than the noise-trained one in this particular evaluation setup.

### 3. Latency difference

Per-sample runtime was drastically different between simulation and hardware:

- statevector: about `0.0007 s/sample`
- IQM: about `0.207-0.208 s/sample`

That makes IQM roughly:

- `289x` slower than statevector for the `ideal` model
- `297x` slower than statevector for the `noise` model

This is expected for real-device execution, but it is important for planning larger experiments.

### 4. Fold-to-fold behavior

Fold winners were mixed, which means the ranking between `ideal` and `noise` is not uniform across all splits.

On `StatevectorEstimator`:

- `ideal` won folds `1` and `5`
- `noise` won folds `2`, `3`, and `4`

On IQM at `512` shots:

- `ideal` won folds `1`, `2`, and `3`
- `noise` won folds `4` and `5`

So while `ideal` is stronger on average, `noise` still wins on specific folds and should not be treated as uniformly worse.

### 5. Stability

The most stable configuration by accuracy standard deviation was:

- `ideal` on IQM: `std = 0.0127`

This is noticeably lower than:

- `ideal` on statevector: `0.0272`
- `noise` on statevector: `0.0374`
- `noise` on IQM: `0.0382`

That means the `ideal` model on hardware was not only competitive, but also the most consistent across the five folds in this run.

## Key Conclusions

1. The `ideal` model is the best overall choice in this experiment.
2. The hardware run at `512` shots preserves the `ideal` model's F1 very well, even though accuracy drops slightly.
3. The `noise` model does not show an average advantage over the `ideal` model, either in simulation or on hardware.
4. IQM execution introduces a large time cost, so hardware sweeps should be used selectively.
5. Because fold-level winners vary, future comparisons should continue to report fold-wise results and not only global averages.

## Recommended Next Steps

1. Repeat the same all-fold evaluation for `best` checkpoints, not only `final`.
2. Test more shot counts, for example `128`, `256`, `512`, and `1024`, to see whether the hardware gap narrows or stabilizes.
3. Add a compact table of `IQM - Statevector` deltas per fold to identify which folds are most hardware-sensitive.
4. Compare the all-fold results against the summary from `cross_validation/raport.md` to check whether hardware inference changes the main conclusions from cross-validation training.

