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

## D3 - Mon 23 Nov 2026 (started early) - Supervised challenger

**Done**
- `models/tune.py` generalised to model families with per-family simplicity order for the one-SE
  rule; HistGradientBoosting added to the registry (early stopping off: its random internal
  split would leak neighbouring segments).
- `evaluation/compare.py`: paired comparison on identical folds, Nadeau-Bengio corrected t-test
  -> `reports/comparisons.json`. Per-fold scores saved in `data/processed/oof/*_folds.csv`.
- `models/explain.py`: SHAP from fold models on held-out segments ->
  `data/processed/shap/hgb_oof.csv`; figures 14-17 (global bar, beeswarm, TP/FN waterfalls);
  figures 18-20 (PR, calibration, paired folds).
- `notebooks/03_gradient_boosting.ipynb` (executed). Tests: one-SE rule, corrected t-test,
  SHAP additivity + out-of-sample coverage (23 tests).

**Evidence**
- HGB grid (leaves {4,8,16,31} x lr {0.03,0.1}): best 0.930; one-SE choice 4 leaves, lr 0.03.
- hgb PR-AUC 0.919 ± 0.060 vs logreg 0.878 ± 0.078; Brier 0.052 vs 0.060; at policy threshold
  both ~0.82 precision / ~0.82 recall.
- Paired folds: hgb better in 11/15, mean +0.040. Naive t-test p = 0.034; corrected p = 0.30.
- SHAP drivers: n_peaks >> diff2_var > diff2_peaks > var ratios; channel contribution small.
- Errors: FN concentrated in CADC0872 (22) and CADC0892 (15, recall 0.44), typically single-peak;
  FP concentrated in CADC0894 (26 vs 13 TP), long multi-peak normal segments.

**Decisions**
- hgb = primary supervised model (better point estimate, variance, calibration; exact SHAP).
  logreg = interpretable baseline. Report states the CV difference is not significant.
- Both go to the frozen test once, after Day 4.

## D4 - Tue 24 Nov 2026 - Novelty track and frozen test

**Done (before test evaluation)**
- Isolation Forest and PCA reconstruction (custom `PCAReconstruction`) in the registry, fit on
  train-normal segments only, quantile-scaled numeric features, no channel input.
- Grids: iforest max_samples {128,256,512} x max_features {0.5,1}; pca n_components {1..10}.
  Labels used only to score folds (hyperparameter + threshold choice), never to fit.

**Evidence (episode-grouped CV, train split)**
- pca_recon (3 comps, one-SE): PR-AUC 0.850 ± 0.065; U-shaped in n_components (1: 0.71,
  3: 0.85, 10: 0.72): too few comps cannot model normal behaviour, too many reconstruct anomalies.
- iforest (128 samples, one-SE): PR-AUC 0.611 ± 0.160.
- Corrected paired t-tests: hgb > pca_recon (14/15 folds, p = 0.034); pca_recon vs logreg tie
  (p = 0.49).
- Complementarity: pca_recon flags 17 of hgb's 56 OOF misses; on CADC0892 recall 0.63 vs 0.44
  but 28 FP vs 5. iforest flags all 131 normal CADC0892 segments and 79 on CADC0884 (zero
  anomalies): channel-unaware novelty models confuse "unusual channel scale" with anomaly.
- `contamination` is not tuned: it only sets sklearn's internal predict() cut-off, which is never
  used; scores + our threshold policy determine every reported metric.

**PRE-REGISTRATION (written before the frozen test is scored)**
- Candidates: prevalence, channel_only, logreg, hgb, pca_recon, iforest (final fits on full
  train split; thresholds fixed from train OOF predictions).
- Primary model: **hgb**. Secondary "second-opinion" score for the dashboard: **pca_recon**.
- Test reported once, with 95% episode-cluster bootstrap intervals, per channel, and on the
  subset of test segments that do not touch a training segment.
- No model, hyperparameter or threshold will be changed after seeing test results; any later
  change would be reported as post-hoc.

**Frozen test results (run once: 2026-11-24 15:28 UTC, lock in `reports/test_lock.json`)**

| model | test PR-AUC [95% episode bootstrap] | precision | recall | MCC |
|---|---|---|---|---|
| prevalence | 0.214 | 0.21 | 1.00 | 0.00 |
| channel_only | 0.317 [0.171, 0.524] | 0.23 | 1.00 | 0.14 |
| logreg | 0.940 [0.893, 0.973] | 0.89 | 0.84 | 0.83 |
| **hgb** | **0.962 [0.941, 0.985]** | **0.92** | **0.86** | **0.86** |
| pca_recon | 0.882 [0.813, 0.950] | 0.76 | 0.82 | 0.73 |
| iforest | 0.712 [0.543, 0.851] | 0.56 | 0.82 | 0.57 |

- Ranking identical to CV. Every model scores higher on test than in grouped CV; hgb test (0.962)
  ~= Day-1 random-CV estimate (0.967) -> the interleaved test split is optimistic, as predicted.
  Grouped CV (0.919) is reported as the generalisation estimate, test for benchmark comparability.
- 56 isolated test segments (touch no train segment; 20 anomalies): hgb PR-AUC 0.995. Small,
  unrepresentative subset (36% anomalous) -> no collapse away from neighbours, nothing stronger.
- CV weak spot replicates: CADC0894 test PR-AUC 0.75, precision 0.50.

**Decisions**
- Final: hgb primary, pca_recon second-opinion score in dashboard, logreg interpretable baseline.
- Future work (not in scope): channel-normalised / per-channel novelty models; temporal split.

**Next (D5)**
- Streamlit dashboard on saved artifacts: overview, data health, segment explorer, model
  comparison, why-flagged (SHAP incl. test segments), review-queue CSV export.

## D5 - Tue 24 Nov 2026 - Decision-support dashboard

**Done**
- `satelliteguard.dashboard.extract` -> `data/processed/dashboard/` (2.7 MB, committed):
  `segments.csv` (all 2,123 segments: metadata, hgb/logreg/pca scores + flags, PCA novelty
  percentile), `shap.csv` (held-out SHAP for train, final-model SHAP for test), `features.csv`,
  `traces.csv.gz` (raw samples, seconds since segment start).
- `satelliteguard.dashboard.queries`: review queue, SHAP top-k (additive, "other features"
  bucket), PR curve, confusion at threshold, episode context. Unit-tested.
- `app/streamlit_app.py` + 5 pages (overview, review queue, segment explorer, model
  comparison, data health). Tests: extract contract (every segment once, test flags identical to
  `test_predictions.csv`, train scores identical to OOF, SHAP additive, test SHAP logit
  reproduces final-model probability) and a headless AppTest run of every page (39 tests).

**Decisions**
- The app never fits or unpickles a model: it reads extracts and `reports/*.json`. No sklearn
  version coupling at serve time; runs without the raw download (Streamlit Cloud-ready).
- Train scores shown are out-of-fold (mean of 3 repeats); train SHAP comes from the first-repeat
  fold model, so its implied probability can differ slightly - stated on the page.
- Novelty score shown as a percentile among train-normal OOF scores (interpretable "more
  unusual than X% of normal segments") rather than raw reconstruction error.
- Threshold sliders default to the pre-registered policy; any other value shows an
  "exploratory" warning. Reported results never use slider values.
- Labels can be hidden in the queue/explorer to view it as an operator would.
- Wording: "anomalous segment", "benchmark label", "for review"; a scope note on every page.

**Next (D6)**
- Technical report (PDF) incl. dashboard screenshots, model card, clean-environment
  reproducibility dry run.

## D6 - Wed 25 Nov 2026 - Report, model card, reproducibility

**Done**
- Technical report `reports/technical_report.pdf` (32 pages; Markdown copy alongside), following
  the proposal's outline: problem, data and label semantics, leakage, methods, results (CV, paired
  tests, calibration, novelty, SHAP/error analysis, frozen test, per channel), dashboard
  (architecture, provenance, user flows, annotated screenshots), reproducibility, limitations.
- Report is generated: `docs/report/technical_report.md` is a template whose metrics are
  `{{ path:fmt }}` placeholders resolved from `reports/*.json` by `python -m satelliteguard.report`
  (unresolved placeholder = build error; tested). Markdown -> HTML (mistune) -> PDF via headless
  Chrome DevTools `printToPDF` with page numbers.
- `scripts/capture_dashboard.py`: light-theme Streamlit + headless Chrome; waits until every
  Plotly chart has drawn, captures full pages, splits tall pages at blank bands for A4.
- `docs/model_card.md` (intended / out-of-scope use, disaggregated test results, failure modes,
  safety considerations).
- `requirements-lock.txt` (exact versions; Windows-only deps marked) and `scripts/reproduce.py`
  (full pipeline + value-by-value diff of metrics/comparisons/leakage JSON vs the commit).
- 41 tests; CI lints `scripts/` too.

**Reproducibility dry run (fresh clone, new venv from `requirements-lock.txt`)**
- First attempt: the Zenodo *file* download failed TLS verification on this network ("hostname
  mismatch" for zenodo.org) although the API metadata call succeeded; not a project defect, and
  certificate checks were not disabled. Rerun with the raw CSVs copied in: `download` still
  verifies them against the checksums Zenodo's API publishes before anything else runs.
- Result (53 min end to end): **all 1,077 values match** (metrics 992/992, comparisons 66/66,
  leakage check 19/19; tol 1e-6). OOF scores, SHAP, test predictions, segment index, tuning
  grid and dashboard extracts are byte-identical; 39/39 tests pass in the clone (the 2 report
  tests were added after the cloned commit).

**Decisions**
- Report numbers come only from the JSON reports, so report, dashboard and README cannot drift.
- Dashboard screenshots are regenerated by script, not taken by hand.
- Author shown as the GitHub handle; name/roll number to be added by the author if required.

## D7 - Wed 25 Nov 2026 (started early) - Final consistency pass and release

**Done**
- CI green on every commit (GitHub Actions, Ubuntu, Python 3.11) - a second OS/Python combination
  besides the Windows/3.13 environment that produced the results.
- `tests/test_docs_consistency.py`: every hand-written number in the README results table and
  the model card (CV, test PR-AUC + CI, precision/recall, Brier, per-channel, threshold) is checked
  against `reports/metrics.json` (17 tests; 58 in total). The report itself is generated.
- README: CI badge, "For reviewers" guide, corrected Approach section (both supervised models
  are *unweighted*; an earlier line wrongly said class-weighted), Limitations section, complete
  step-by-step commands, data-licence note, dashboard screenshot.
- Version 1.0.0 (`pyproject.toml`), `CHANGELOG.md`; MIT licence for the code (data stays CC-BY-4.0).

**Decisions**
- No model, threshold or result changed on Day 7; documentation and tests only.
- Release tag `v1.0.0` marks the submission commit.
