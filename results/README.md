# results/

Artifacts produced by `src/main.py` and `analysis/check_label_distribution.py`.

Each training run writes, keyed by `RUN_ID` (e.g. `gru_h21_none`):

- `summary_<RUN_ID>.json` — config snapshot, test metrics, VSN top features
- `diagnostics_<RUN_ID>.json` — prediction distribution, confusion matrix,
  confidence-bucket accuracy
- `figures/feature_interaction_<RUN_ID>.png`
- `figures/attention_dynamics_<RUN_ID>_<TICKER>.png`

`label_distribution_sensitivity.csv` is written by the analysis script.

Per-timestep VSN weight series (`attention_timeseries_*.csv`) are also emitted
at run time but kept out of the repo for size; regenerate them by running training.

Model checkpoints (`*.pth`) are git-ignored; regenerate by running training.
