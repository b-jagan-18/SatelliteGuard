# Decision log

One entry per day: what was done, evidence, risks, next decision.

## D0 - Mon 23 Nov 2026 - Kickoff

**Done**
- Repository scaffolded (src package, configs, tests, CI).
- Data download script with Zenodo checksum verification + `data/manifest.json`.
- Data-contract validator (`python -m satelliteguard.data.validate`) and tests.

**Decisions**
- Dataset: OPSSAT-AD, Zenodo record 15108715 (CC-BY-4.0). Raw files are re-downloadable and
  not committed; the manifest (hashes, sizes, row counts) is committed.
- Experiment tracking: a single `reports/metrics.json` instead of MLflow; manifest instead of DVC.
- Publisher split (`train` flag) is the frozen final test, scored exactly once.

**Evidence (data audit, all 12 checks PASS)**
- Checksums match Zenodo; split counts match publisher (1,273/321 train, 416/113 test).
- Schema gate resolved: raw `label` column is the constant string `"anomaly"` in all 303,493 rows
  (a column name, not a class label). Numeric `anomaly` is the only target; `label` is excluded.
- Segment length varies widely: median 70 samples, range 8-1,040.
- Anomaly rate varies strongly by channel: CADC0890 = 78.6%, CADC0884 = 0%.
  **Risk:** a model could learn "channel ⇒ anomaly" rather than signal shape. Report a
  channel-only baseline so feature models must beat it, and show per-channel metrics.

## D1 - Mon 23 Nov 2026 (started early) - Data contract and EDA

**Done**
- `notebooks/01_eda.ipynb` (executed): provenance, contract, class balance, lengths, features,
  correlation, drift, raw-trace gallery, neighbour-leakage analysis.
- Figure module `satelliteguard.eda` -> `reports/figures/01-07*.png`.
- Episode grouping `satelliteguard.data.episodes` -> `data/processed/segment_index.csv`.
- Leakage experiment `satelliteguard.evaluation.leakage_check` -> `reports/leakage_check.json`.

**Evidence**
- Segments are back-to-back slices of continuous telemetry (median gap = one sampling step).
  243 contiguous episodes (gap <= 10 s); median 3 segments, max 220.
- Anomalies cluster: P(anom | prev anom) = 0.37 vs 0.16 after a normal segment.
- Publisher split is interleaved: 473 / 529 test segments (89%) touch a train segment;
  117 episodes contain both train and test segments.
- Same fixed HistGB model on train split: random CV PR-AUC 0.967 ± 0.013 vs episode-grouped
  0.931 ± 0.046 -> random CV is optimistic by ~0.04 and hides variance.
- Channel-only baseline PR-AUC 0.25-0.28 (prevalence 0.20): channel identity is not a shortcut.
- No train/test feature drift (all KS < 0.07, none p < 0.01).
- Strongest single features: n_peaks, kurtosis, smoothed peak counts. Several near-duplicate
  features (var/std, var_div_len/var_div_duration).

**Decisions**
- All model selection, tuning and thresholds use StratifiedGroupKFold on episodes.
- Publisher test split stays the frozen final test (comparability with published benchmark);
  report states it shares episodes with train, so it is optimistic about generalisation.
- Channel-only baseline is a reference row in every results table.
- Logistic baseline: regularised; coefficients not interpreted as independent effects.

## D2 - Mon 23 Nov 2026 (started early) - Reproducible baseline

**Done**
- `features.py` (modelling table, quantile/standard/none preprocessors, channel one-hot inside
  pipeline), `evaluation/metrics.py` (PR-AUC, ROC-AUC, recall/precision@top-25%, Brier, P/R/F1,
  MCC, balanced accuracy; threshold policy), `evaluation/cv.py` (grouped repeated CV),
  `models/registry.py`, `models/tune.py`, `models/train.py`, `evaluation/plots.py`.
- `notebooks/02_baselines.ipynb` (executed); figures 10-13; `reports/metrics.json` (`cv` section);
  OOF predictions in `data/processed/oof/`; final models in `models/*.joblib`.
- Tests: metric correctness, recall floor, grouped folds never split an episode, OOF covers the
  train split only (19 tests total).

**Evidence (episode-grouped CV, 15 folds, train split only)**

| model | PR-AUC | ROC-AUC | P / R at policy | Brier |
|---|---|---|---|---|
| prevalence | 0.201 ± 0.002 | 0.500 | - | - |
| channel_only | 0.250 ± 0.050 | 0.485 | - | - |
| **logreg** (C=1, quantile, unweighted) | **0.878 ± 0.078** | 0.945 | 0.83 / 0.82 | 0.060 |
| logreg_balanced | 0.873 ± 0.083 | 0.948 | 0.76 / 0.89 | 0.071 |

- Grid C in {0.01..100} x {quantile, standard}: best C=100 (0.889) but within noise; one-SE rule
  selects C=1, quantile.
- class_weight="balanced" degrades calibration (Brier 0.060 -> 0.071) with no PR-AUC gain.
- Logistic coefficients show opposite signs on correlated pairs (duration/len_weighted,
  var/var_div_len) -> not interpretable individually.

**Decisions**
- Primary metric reported as fold mean ± sd; differences < ~1 sd are treated as ties.
- Threshold policy: max F1 s.t. recall >= 0.80, chosen on OOF train predictions (logreg: 0.383).
- Primary logistic model is unweighted (calibrated probabilities shown in dashboard);
  imbalance handled by the threshold, not by re-weighting.
- Frozen test still untouched; it is scored once, after Day 4 model selection.

**Next (D3)**
- HistGradientBoosting (bounded grid, grouped CV); must beat 0.878 by more than fold noise.
- SHAP global + per-segment explanations for the tree model.
