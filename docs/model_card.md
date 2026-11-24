# Model card: SatelliteGuard segment anomaly ranker

Format adapted from Mitchell et al., *Model Cards for Model Reporting* (FAT* 2019).

## Model details

| | |
|---|---|
| **Primary model** | Histogram gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`): 300 trees, 4 leaves/tree, learning rate 0.03, min 20 samples/leaf, L2 = 1.0, unweighted, early stopping **off** |
| **Second-opinion score** | PCA reconstruction error (3 components) on quantile-normalised features, fit on normal training segments only (no labels used in fitting) |
| **Interpretable baseline** | L2 logistic regression (C = 1), quantile-normal scaling, unweighted |
| **Inputs** | 18 publisher summary features per segment (duration, length, mean, variance, std, kurtosis, skew, peak counts, first/second-difference statistics, gap and ratio features) + sampling interval + channel one-hot (9 channels). Novelty models do not use the channel. |
| **Output** | Probability that the segment carries the *anomalous* benchmark label; flag if ≥ **0.389** |
| **Artifacts** | `models/hgb.joblib`, `models/pca_recon.joblib`, `models/logreg.joblib` (scikit-learn 1.9.1; see `requirements-lock.txt`) |
| **Version / date** | v1.0, 24 Nov 2026 (frozen-test run recorded in `reports/test_lock.json`) |
| **Author** | GitHub `b-jagan-18` (solo FOML course project) |
| **License** | Code: see repository. Data: OPSSAT-AD, CC-BY-4.0 |

## Intended use

- **Primary use:** rank telemetry segments from the OPSSAT-AD benchmark so that a human reviewer
  looks at the most unusual segments first (decision support, review queue).
- **Primary users:** students, researchers and analysts studying anomaly detection on spacecraft
  telemetry; course assessors reproducing the benchmark.
- **How to use the output:** treat a flag as "review this segment", read the per-segment SHAP
  explanation and the novelty percentile, and look at the raw trace before drawing conclusions.

## Out-of-scope uses

- **Diagnosing or confirming spacecraft faults.** Labels are segment-level benchmark
  annotations. A flag means "resembles segments annotated as anomalous", not "hardware fault".
- **Inferring subsystem behaviour.** Channel IDs (e.g. `CADC0872`) are opaque. No physical or
  subsystem meaning is claimed.
- **Autonomous action** (commanding, safing, alert escalation without human review).
- **Other missions, channels or sampling regimes** without re-validation. The model has seen only
  9 OPS-SAT channels at 1 s and 5 s sampling.
- **Real-time streaming.** The model scores pre-segmented windows with publisher-computed
  features; segmentation itself is not part of this system.

## Training data

OPSSAT-AD (Ruszczak et al., Zenodo 15108715, v2): 2,123 segments from 9 channels.

| Split | Normal | Anomalous | Use |
|---|---|---|---|
| train (`train=1`) | 1,273 | 321 | fitting, tuning, threshold choice (episode-grouped CV) |
| test (`train=0`) | 416 | 113 | frozen; scored once |

Anomaly rate varies by channel (0% on CADC0884 to 79% on CADC0890). Segments are back-to-back
slices of continuous telemetry, grouped into 243 contiguous episodes (gap ≤ 10 s, same channel).

## Evaluation

**Protocol.** All selection uses 5-fold × 3-repeat StratifiedGroupKFold on episodes
(neighbouring segments never straddle a fold). Hyperparameters use the one-standard-error rule.
The threshold maximises F1 subject to recall ≥ 0.80 on out-of-fold train predictions. The test
split was scored once, after the candidates were pre-registered in `docs/decision_log.md`.
Confidence intervals: 2,000-resample bootstrap over episodes.

**Primary metric:** PR-AUC (average precision). Chance level equals the anomaly rate (≈ 0.21).

| Model | Grouped-CV PR-AUC | Test PR-AUC [95% CI] | Test precision | Test recall | Test Brier |
|---|---|---|---|---|---|
| Gradient boosting (primary) | 0.919 ± 0.060 | **0.962 [0.941, 0.985]** | 0.92 | 0.86 | 0.035 |
| Logistic regression | 0.878 ± 0.078 | 0.940 [0.893, 0.973] | 0.89 | 0.84 | 0.044 |
| PCA reconstruction | 0.850 ± 0.065 | 0.882 [0.813, 0.950] | 0.76 | 0.82 | – |
| Channel-only reference | 0.250 ± 0.050 | 0.317 [0.171, 0.524] | 0.23 | 1.00 | 0.159 |

At the policy threshold the primary model flags 106 of 529 test segments (20%), of which 97 carry
the anomalous label; 16 of 113 anomalous segments are not flagged.

**Which number to trust.** 89% of test segments touch a training segment, so the test split
measures performance on *interleaved* data and is optimistic. It matches a leaky random-CV
estimate (0.967). **Grouped CV (≈ 0.92) is the better estimate for new, unseen periods.**

**Disaggregated (test, primary model).**

| Channel | n | Anomaly rate | PR-AUC | Precision | Recall |
|---|---|---|---|---|---|
| CADC0872 | 132 | 0.24 | 0.971 | 0.90 | 0.88 |
| CADC0873 | 153 | 0.20 | 0.975 | 0.93 | 0.90 |
| CADC0874 | 52 | 0.44 | 0.995 | 1.00 | 0.74 |
| CADC0888 | 64 | 0.19 | 0.976 | 1.00 | 0.92 |
| CADC0892 | 53 | 0.13 | 0.905 | 1.00 | 0.86 |
| **CADC0894** | 33 | 0.15 | **0.754** | **0.50** | 0.80 |
| CADC0884 / 0886 / 0890 | 36 / 4 / 2 | – | single class or too small to rank | | |

Segments that touch no training segment (56; 20 anomalous): PR-AUC 0.995. This subset is small
and unrepresentative, so it shows no collapse away from neighbours but proves nothing stronger.

## Explainability

Per-segment SHAP values (TreeExplainer, log-odds) are computed out of sample: train segments by
the fold model that did not see their episode, test segments by the final model. Globally the
model relies on peak counts (`n_peaks`) and second-difference variance/peaks (spikes and abrupt
changes); channel identity contributes little. SHAP describes **model association, not physical
cause**.

## Known limitations and failure modes

- **CADC0894 false alarms:** long, multi-peak *normal* segments are scored as anomalous
  (precision 0.50 on test, the same pattern in CV).
- **Single-peak anomalies are missed** most often (CADC0872, CADC0892 in CV).
- **Benchmark labels are the ground truth.** Label noise or annotation conventions in OPSSAT-AD
  propagate directly into the model and its metrics.
- **Interleaved test split:** test metrics overstate generalisation to new time periods.
- **Small channels** (CADC0886, CADC0890; 11 and 14 segments) are effectively unvalidated.
- **Channel-unaware novelty models** confuse a channel's unusual normal range with anomaly
  (Isolation Forest flags all normal CADC0892 segments); this is why only PCA is shown, as a
  second opinion.
- **Distribution shift:** no temporal hold-out was available; drift over the mission lifetime is
  untested.

## Ethical and safety considerations

- **Human in the loop.** Outputs are a ranked queue for review, never an automatic decision.
- **Over-reliance risk.** A confident score is not evidence of a fault. The dashboard places a
  scope note on every page and shows the raw trace, the SHAP explanation and an independent
  novelty score so reviewers can disagree with the model.
- **Missed anomalies.** At the default threshold about 14% of anomalous-labelled test segments
  are not flagged. Lower the threshold (higher recall, more alerts) if misses are costly; the
  dashboard's threshold page shows the trade-off.
- **No personal data** is involved; the data are public spacecraft telemetry (CC-BY-4.0).

## Reproducibility

`pip install -r requirements-lock.txt && pip install -e .`, then `python scripts/reproduce.py`
re-runs the full pipeline from the Zenodo download and diffs every reported number against the
committed `reports/*.json`. Seed 42; all transforms are fit inside CV folds.
