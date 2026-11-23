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

**Next (D2)**
- Feature pipeline + metric module (PR-AUC, recall@budget, MCC, Brier) writing `reports/metrics.json`.
- Calibrated logistic regression baseline under grouped CV; threshold policy from CV predictions.
