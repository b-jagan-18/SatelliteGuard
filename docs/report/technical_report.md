<div class="title-page">
<div class="kicker">Foundations of Machine Learning · Course project · Technical report</div>
<h1>SatelliteGuard</h1>
<div class="subtitle">Leakage-safe anomaly detection and review prioritisation for ESA OPS-SAT telemetry</div>
<div class="kpis">
<div class="kpi"><b>{{test.hgb.overall.pr_auc:.3f}}</b><span>frozen-test PR-AUC, gradient boosting (95% CI {{test.hgb.bootstrap.pr_auc_ci95.0:.2f}}–{{test.hgb.bootstrap.pr_auc_ci95.1:.2f}})</span></div>
<div class="kpi"><b>{{cv.hgb.cv_folds.pr_auc_mean:.3f}}</b><span>episode-grouped CV PR-AUC: the honest estimate for unseen periods</span></div>
<div class="kpi"><b>{{test.hgb.overall.precision:.2f}} / {{test.hgb.overall.recall:.2f}}</b><span>test precision / recall at the pre-registered threshold</span></div>
<div class="kpi"><b>{{leak.optimism_pr_auc:.3f}}</b><span>PR-AUC optimism of random CV caused by neighbour leakage</span></div>
</div>
<div class="meta">
<b>Author:</b> GitHub <code>b-jagan-18</code> (solo project)<br>
<b>Repository:</b> <a href="https://github.com/b-jagan-18/SatelliteGuard">github.com/b-jagan-18/SatelliteGuard</a><br>
<b>Data:</b> OPSSAT-AD v2, Ruszczak et al., Zenodo 15108715 (CC-BY-4.0)<br>
<b>Frozen test scored once:</b> {{lock.evaluated_utc}} (lock file <code>reports/test_lock.json</code>)<br>
<b>Date:</b> November 2026
</div>
<div class="callout scope"><b>Scope.</b> SatelliteGuard ranks telemetry <i>segments for review</i>. A flag means a segment
resembles segments annotated as anomalous in a public benchmark. It does <b>not</b> diagnose or confirm spacecraft
hardware faults, and channel identifiers carry no claimed subsystem meaning.</div>
</div>

## Abstract

<div class="abstract">

Spacecraft operators cannot inspect every stretch of telemetry, so the practical question is *which segments deserve
attention first, with what evidence and uncertainty*. SatelliteGuard answers it on OPSSAT-AD, a public benchmark of
2,123 labelled telemetry segments from nine channels of ESA's OPS-SAT satellite. The central methodological finding is
**neighbour leakage**: segments are back-to-back slices of continuous signals, and the publisher's train/test split
interleaves them, so 89% of test segments touch a training segment. A random cross-validation (CV) overstates PR-AUC by
{{leak.optimism_pr_auc:.3f}} relative to a CV that keeps contiguous *episodes* together. All model selection, tuning
and threshold choice therefore use repeated episode-grouped CV on the training split, and the publisher's test split is
scored exactly once after pre-registration.

Four model families are compared against two reference baselines. Histogram gradient boosting is the strongest
(grouped-CV PR-AUC {{cv.hgb.cv_folds.pr_auc_mean:.3f}} ± {{cv.hgb.cv_folds.pr_auc_std:.3f}}; frozen test
{{test.hgb.overall.pr_auc:.3f}}, 95% episode-bootstrap CI [{{test.hgb.bootstrap.pr_auc_ci95.0:.3f}},
{{test.hgb.bootstrap.pr_auc_ci95.1:.3f}}], precision {{test.hgb.overall.precision:.2f}}, recall
{{test.hgb.overall.recall:.2f}}). After correcting for overlapping training sets, its advantage over logistic
regression is not significant (p = {{cmp.hgb_vs_logreg_pr_auc.p_corrected:.2f}}). Its advantage over a label-free
PCA-reconstruction novelty model is significant (p = {{cmp.hgb_vs_pca_recon_pr_auc.p_corrected:.3f}}). The PCA model
still catches a share of the boosting model's misses and is shown as a second opinion. Out-of-sample SHAP
explanations show the model relies on peak counts and second-difference statistics rather than channel identity. The
results are delivered through a Streamlit dashboard (review queue, per-segment explanations, model comparison, data
health) that runs only on versioned, saved artifacts. A clean-environment re-run reproduces every reported number.

</div>

<div class="toc">

**Contents**

1. Problem and motivation
2. Data and label semantics
3. Neighbour leakage
4. Methods
5. Results
6. Dashboard
7. Reproducibility and engineering
8. Limitations and next steps
9. Conclusion
10. References
11. Appendices

</div>

<div class="pagebreak"></div>

## 1. Problem and motivation

Satellite operations teams receive continuous housekeeping telemetry from many sensors. Most of it is nominal, and
reviewer attention is the scarce resource. A useful machine-learning system therefore does not need to *explain*
faults. It needs to **rank** stretches of telemetry so that the unusual ones are seen early. It should state how
much the ranking can be trusted, and it should show why each item was ranked high.

This project builds such a ranking system on OPSSAT-AD [1, 2], a benchmark released by the OPS-SAT team in which
telemetry has been cut into segments and each segment carries a binary *anomalous / normal* annotation. The decision
question, fixed in the project proposal, is:

> *Which telemetry segments deserve early operator investigation, with what evidence and uncertainty?*

Three requirements follow from this framing and shape every later choice:

1. **Honest performance estimates.** A reviewer must know how the ranking will behave on *new* telemetry, not just on
   the benchmark split. This leads to the leakage analysis of Section 3 and the grouped validation protocol of
   Section 4.
2. **Operational outputs.** A ranked queue, a threshold that turns scores into a manageable number of alerts, and
   per-item evidence (raw trace, explanation, independent second opinion).
3. **A claim boundary.** Labels are segment-level benchmark annotations, not confirmed hardware faults, and channel
   IDs are opaque. The report and the dashboard use the terms *anomalous segment* and *benchmark label* throughout, and
   never attribute an anomaly to a subsystem.

**Contributions.** (i) A quantified demonstration that the publisher split is affected by neighbour leakage, and a
grouped protocol that corrects for it. (ii) A bounded, pre-registered comparison of supervised and novelty-detection
models with corrected significance tests and a once-only frozen-test evaluation with episode-level confidence
intervals. (iii) Out-of-sample explanations and error analysis by channel. (iv) A decision-support dashboard and a
reproducible package whose report numbers are generated from a single metrics file.

<div class="pagebreak"></div>

## 2. Data and label semantics

### 2.1 Provenance and verification

The data are OPSSAT-AD version 2 (Zenodo record 15108715, CC-BY-4.0) [1]. Two files are used:

| File | Rows | Contents |
|---|---|---|
| `segments.csv` | 303,493 | raw samples: channel, timestamp, value, sampling interval, segment id, `anomaly`, `train` |
| `dataset.csv` | 2,123 | one row per segment: 18 publisher summary features, channel, sampling, `anomaly`, `train` |

The download script verifies each file against the checksums published by Zenodo and records hashes, sizes and row
counts in `data/manifest.json`. A data-contract validator then runs twelve checks that gate every later step; all
twelve pass. Among them:

- one channel, label and split per segment
- identical segment ids, labels and splits in both files
- publisher split counts reproduced exactly
- timestamps parseable and monotonic within each segment
- no missing values

**Label semantics.** The raw file contains a column called `label` whose value is the constant string `"anomaly"` in
all 303,493 rows. It is a column name, not a class label. The numeric `anomaly` column is the only target, and
`label` is excluded from all features. Positive class = 1 = *anomalous segment*.

| Split | Normal | Anomalous | Total | Anomaly rate |
|---|---|---|---|---|
| train (`train = 1`) | 1,273 | 321 | 1,594 | {{cv.hgb.cv_oof_at_threshold.prevalence:.3f}} |
| frozen test (`train = 0`) | 416 | 113 | 529 | {{test.hgb.overall.prevalence:.3f}} |

![Class balance](figures/01_class_balance.png)

*Figure 1. Class balance by split and channel. Anomaly rates range from 0% (CADC0884) to 79% (CADC0890); two channels have fewer than 15 segments.*

### 2.2 Channels and features

The nine channels fall into two sampling regimes: three channels at 1 s and six at 5 s. Segment length varies from 8 to
1,040 samples (median 70). The 18 publisher features summarise each segment's length and duration, moments (mean,
variance, kurtosis, skew), peak counts on the raw and smoothed signal, first and second differences, gaps and
length-normalised variance. We add the sampling interval as a numeric acquisition property.

**Channel caveat.** Anomaly rate differs strongly by channel, so a model could score well by learning *which channel
is this?* instead of *what does the signal look like?*. A **channel-only baseline** that predicts each channel's
training anomaly rate is therefore a reference row in every table. Every feature model must beat it, and results are
also reported per channel.

![Feature distributions](figures/04_feature_distributions.png)

*Figure 2. Distribution of the most separating features by class (train split). Peak counts and kurtosis separate the classes best on their own; several features are near-duplicates (variance and standard deviation, the two variance ratios).*

A univariate screen (Appendix B) ranks `n_peaks` first (single-feature ROC-AUC 0.84 on train). Train and test feature
distributions show no drift: every two-sample Kolmogorov–Smirnov statistic is below 0.07 and none is significant at
the 1% level.

## 3. Neighbour leakage

### 3.1 Segments are slices of continuous signals

Sorting segments by channel and start time shows that consecutive segments usually abut: the median gap between a
segment and its predecessor is a single sampling step. We define a **contiguous episode** as a maximal run of segments
on the same channel with gaps of at most 10 s. There are **243 episodes** (median 3 segments, largest 220). Anomalies
cluster in time: P(anomalous | previous segment anomalous) = 0.37 versus 0.16 after a normal segment.

The publisher's split is assigned per segment, not per episode, so it **interleaves** train and test inside
episodes: 473 of 529 test segments (89%) touch a training segment, and 117 episodes contain both.

![Segment timeline](figures/02_segment_timeline.png)

*Figure 3. Segments in time order within each channel, coloured by benchmark label (top) and publisher split (bottom). Black ticks mark episode starts. Train and test segments alternate inside the same episodes.*

### 3.2 How large is the effect?

The same fixed gradient-boosting model was evaluated on the training split with two 5-fold × 3-repeat protocols that
differ only in how folds are formed:

| Protocol | Model PR-AUC | Model ROC-AUC | Channel-only PR-AUC |
|---|---|---|---|
| Random stratified folds | {{leak.random.model_pr_auc.mean:.3f}} ± {{leak.random.model_pr_auc.std:.3f}} | {{leak.random.model_roc_auc.mean:.3f}} | {{leak.random.channel_only_pr_auc.mean:.3f}} |
| Episode-grouped folds | {{leak.grouped.model_pr_auc.mean:.3f}} ± {{leak.grouped.model_pr_auc.std:.3f}} | {{leak.grouped.model_roc_auc.mean:.3f}} | {{leak.grouped.channel_only_pr_auc.mean:.3f}} |

Random folds overstate PR-AUC by **{{leak.optimism_pr_auc:.3f}}** and understate its variability by a factor of more
than three. Near-identical neighbours on both sides of a random split let the model partly memorise local signal
shape. The publisher's test split has the same structure. **Prediction (made on Day 1, before any test scoring):**
frozen-test scores will look like random-CV scores, not grouped-CV scores. Section 5.6 confirms this.

**Consequences for the protocol.**

- Every model choice uses episode-grouped CV on the training split.
- The publisher test split is kept as the frozen final test, so results stay comparable with the published benchmark.
- The report states that the test split is optimistic about generalisation to new periods.

## 4. Methods

### 4.1 Preprocessing and leakage controls

All transforms are scikit-learn objects inside a `Pipeline` and are therefore fit on the training fold only.
Identifiers, the target and the split flag never enter the feature matrix.

- **Linear and novelty models:** quantile transformation of numeric features to a normal distribution. The table
  contains heavy tails and tiny magnitudes (variances around 10⁻¹²), and rank-based scaling handles both.
- **Trees:** raw features.
- **Channel:** one-hot encoded for supervised models; *excluded* from novelty models, which are meant to describe
  signal shape.

### 4.2 Model matrix

| Track | Model | What it learns | Key settings |
|---|---|---|---|
| Reference | Prevalence | constant score | – |
| Reference | Channel only | training anomaly rate of the segment's channel | – |
| Supervised | Logistic regression | linear log-odds in scaled features | L2, C chosen by CV, unweighted |
| Supervised | Gradient boosting | additive trees on raw features | histogram GBT, 300 trees, early stopping **off** |
| Novelty | PCA reconstruction | low-dimensional subspace of *normal* segments | fit on normal training segments only |
| Novelty | Isolation Forest | how easily a point is isolated by random splits | fit on normal training segments only |

Two non-default choices matter. First, scikit-learn's early stopping for gradient boosting holds out a *random* 10%
of the training fold, which would re-introduce neighbour leakage inside every fold, so it is disabled and the number of
trees is fixed. Second, the logistic model is **unweighted**. With `class_weight="balanced"` PR-AUC did not improve
({{cv.logreg_balanced.cv_folds.pr_auc_mean:.3f}} vs {{cv.logreg.cv_folds.pr_auc_mean:.3f}}) but calibration got worse
(Brier {{cv.logreg_balanced.cv_oof_at_threshold.brier:.3f}} vs {{cv.logreg.cv_oof_at_threshold.brier:.3f}}).
Class imbalance is handled by the threshold instead.

The **PCA reconstruction** score is the squared distance between a segment's scaled feature vector *x* and its
projection onto the top *k* principal directions of normal training segments:
score(*x*) = ‖(*x* − μ) − *V*ₖ*V*ₖᵀ(*x* − μ)‖². Segments unlike anything normal reconstruct poorly.

### 4.3 Validation protocol

`StratifiedGroupKFold` with episodes as groups, 5 folds × 3 repeats (15 train/validation pairs, seed 42), on the
training split only. Stratification keeps each fold near the 20% anomaly rate, and grouping guarantees that no
episode is split across a fold boundary (this is unit-tested). Out-of-fold (OOF) scores are averaged over the three
repeats, giving one honest score per training segment. These scores are used for threshold choice, error analysis
and the dashboard.

**Metrics.**

- **Primary:** PR-AUC (average precision), because positives are about 20% and the task is ranking. Chance level
  equals the anomaly rate.
- **Secondary:** ROC-AUC; recall and precision in the top 25% of scores (an alert budget); and, at the operating
  threshold, precision, recall, F1, Matthews correlation coefficient (MCC), balanced accuracy and Brier score
  (probabilistic models only).
- CV results are reported as mean ± standard deviation over the 15 folds. Differences smaller than about one standard
  deviation are treated as ties unless a paired test says otherwise.

### 4.4 Bounded tuning and the one-standard-error rule

Each family has a small, pre-declared grid (Appendix A). The highest mean PR-AUC on 15 folds is a noisy estimate, so
we apply the **one-standard-error rule**: among configurations whose mean is within one standard error of the best,
pick the simplest. Simplicity is ordered per family: smaller C; fewer leaves, then smaller learning rate; fewer
components; smaller subsamples.

### 4.5 Threshold policy

A ranking becomes a review queue only once a threshold is fixed. Missing an anomalous segment is assumed costlier
than an extra review, so the policy is: **maximise F1 subject to recall ≥ 0.80**, chosen on OOF training scores.
This yields a probability threshold of {{cv.hgb.threshold_policy.threshold:.3f}} for gradient boosting and
{{cv.logreg.threshold_policy.threshold:.3f}} for logistic regression. Thresholds are frozen before the test split is
touched.

![Threshold trade-off](figures/12_cv_threshold_tradeoff.png)

*Figure 4. Review-queue trade-off for the logistic baseline (OOF). Precision, recall, F1 and the fraction of segments flagged as the threshold moves; the vertical line marks the policy threshold.*

### 4.6 Comparing models: corrected paired t-test

All models are evaluated on identical folds, so per-fold differences are paired. A naive paired t-test over the 15
folds is over-confident, because training sets overlap heavily between folds and the per-fold differences are
correlated. We use the **Nadeau–Bengio corrected resampled t-test** [5]. It inflates the variance of the mean
difference from σ²/*n* to (1/*n* + *n*ₜₑₛₜ/*n*ₜᵣₐᵢₙ)·σ², with *n*ₜₑₛₜ/*n*ₜᵣₐᵢₙ = 1/4 for 5-fold CV.

### 4.7 Frozen-test protocol

Before the test split was opened, the decision log recorded:

- the six candidate models
- the primary model (gradient boosting)
- the second-opinion score (PCA reconstruction)
- the fixed thresholds
- the subgroup analyses to be reported
- a commitment that no model, hyperparameter or threshold would change afterwards

Final models were refit on the full training split. The evaluation script writes a lock file with timestamp and
commit, and refuses to run again (a test asserts this).

**Uncertainty.** Test segments within an episode are correlated, so a naive bootstrap over segments would give
intervals that are too narrow. 95% intervals come from **2,000 bootstrap resamples of whole episodes**, recomputing
each metric at the fixed threshold.

### 4.8 Explainability

The gradient-boosting model is explained with exact TreeSHAP [6] in log-odds units. To keep explanations out of
sample, each training segment is explained by the fold model that did not see its episode (first CV repeat), and each
test segment by the final model. One-hot channel columns are summed into a single *channel* contribution. SHAP values
are additive (base value + Σ contributions = model log-odds, unit-tested) and describe **how the model uses features,
not physical causes**.

## 5. Results

### 5.1 Cross-validated comparison

| Model | Track | PR-AUC (mean ± sd, 15 folds) | ROC-AUC | Precision @ policy | Recall @ policy | MCC | Brier |
|---|---|---|---|---|---|---|---|
| Prevalence | reference | {{cv.prevalence.cv_folds.pr_auc_mean:.3f}} ± {{cv.prevalence.cv_folds.pr_auc_std:.3f}} | {{cv.prevalence.cv_folds.roc_auc_mean:.3f}} | – | – | – | – |
| Channel only | reference | {{cv.channel_only.cv_folds.pr_auc_mean:.3f}} ± {{cv.channel_only.cv_folds.pr_auc_std:.3f}} | {{cv.channel_only.cv_folds.roc_auc_mean:.3f}} | – | – | – | – |
| Logistic regression | supervised | {{cv.logreg.cv_folds.pr_auc_mean:.3f}} ± {{cv.logreg.cv_folds.pr_auc_std:.3f}} | {{cv.logreg.cv_folds.roc_auc_mean:.3f}} | {{cv.logreg.cv_oof_at_threshold.precision:.2f}} | {{cv.logreg.cv_oof_at_threshold.recall:.2f}} | {{cv.logreg.cv_oof_at_threshold.mcc:.2f}} | {{cv.logreg.cv_oof_at_threshold.brier:.3f}} |
| **Gradient boosting** | supervised | **{{cv.hgb.cv_folds.pr_auc_mean:.3f}} ± {{cv.hgb.cv_folds.pr_auc_std:.3f}}** | {{cv.hgb.cv_folds.roc_auc_mean:.3f}} | {{cv.hgb.cv_oof_at_threshold.precision:.2f}} | {{cv.hgb.cv_oof_at_threshold.recall:.2f}} | {{cv.hgb.cv_oof_at_threshold.mcc:.2f}} | **{{cv.hgb.cv_oof_at_threshold.brier:.3f}}** |
| PCA reconstruction | novelty | {{cv.pca_recon.cv_folds.pr_auc_mean:.3f}} ± {{cv.pca_recon.cv_folds.pr_auc_std:.3f}} | {{cv.pca_recon.cv_folds.roc_auc_mean:.3f}} | {{cv.pca_recon.cv_oof_at_threshold.precision:.2f}} | {{cv.pca_recon.cv_oof_at_threshold.recall:.2f}} | {{cv.pca_recon.cv_oof_at_threshold.mcc:.2f}} | – |
| Isolation Forest | novelty | {{cv.iforest.cv_folds.pr_auc_mean:.3f}} ± {{cv.iforest.cv_folds.pr_auc_std:.3f}} | {{cv.iforest.cv_folds.roc_auc_mean:.3f}} | {{cv.iforest.cv_oof_at_threshold.precision:.2f}} | {{cv.iforest.cv_oof_at_threshold.recall:.2f}} | {{cv.iforest.cv_oof_at_threshold.mcc:.2f}} | – |

*Table 1. Episode-grouped CV on the training split. Precision, recall, MCC and Brier are computed on the pooled out-of-fold scores at each model's policy threshold.*

- **The features carry the signal; channel identity does not.** The channel-only baseline barely beats prevalence
  ({{cv.channel_only.cv_folds.pr_auc_mean:.3f}} vs {{cv.prevalence.cv_folds.pr_auc_mean:.3f}}). Every feature model
  is far above both.
- **Gradient boosting** has the best mean, the smallest spread and the best calibration among supervised models.
- **PCA reconstruction**, fit without any labels, is statistically tied with the supervised logistic model (mean
  paired difference {{cmp.pca_recon_vs_logreg_pr_auc.mean_diff:+.3f}} PR-AUC, Table 2).
- **Isolation Forest** is weak and unstable (sd {{cv.iforest.cv_folds.pr_auc_std:.3f}}).

![PR curves, all tracks](figures/25_cv_pr_all_tracks.png)

*Figure 5. Out-of-fold precision–recall curves for all tracks (train split, scores averaged over three CV repeats); the legend gives each model's per-fold mean PR-AUC.*

### 5.2 Is the best model really better? Paired comparisons

| Comparison (A vs B) | Mean ΔPR-AUC | Folds A better | Naive p | Corrected p | Verdict |
|---|---|---|---|---|---|
| Gradient boosting vs logistic | {{cmp.hgb_vs_logreg_pr_auc.mean_diff:+.3f}} | {{cmp.hgb_vs_logreg_pr_auc.folds_a_better}} / 15 | {{cmp.hgb_vs_logreg_pr_auc.p_naive:.3f}} | **{{cmp.hgb_vs_logreg_pr_auc.p_corrected:.3f}}** | not significant |
| Gradient boosting vs PCA recon. | {{cmp.hgb_vs_pca_recon_pr_auc.mean_diff:+.3f}} | {{cmp.hgb_vs_pca_recon_pr_auc.folds_a_better}} / 15 | {{cmp.hgb_vs_pca_recon_pr_auc.p_naive:.4f}} | **{{cmp.hgb_vs_pca_recon_pr_auc.p_corrected:.3f}}** | significant at 5% |
| PCA recon. vs logistic | {{cmp.pca_recon_vs_logreg_pr_auc.mean_diff:+.3f}} | {{cmp.pca_recon_vs_logreg_pr_auc.folds_a_better}} / 15 | {{cmp.pca_recon_vs_logreg_pr_auc.p_naive:.3f}} | {{cmp.pca_recon_vs_logreg_pr_auc.p_corrected:.3f}} | tie |

*Table 2. Paired comparisons on identical folds with the Nadeau–Bengio correction.*

The first row is the textbook case for the correction. Boosting wins 11 of 15 folds and a naive t-test calls the gain
significant (p = {{cmp.hgb_vs_logreg_pr_auc.p_naive:.3f}}). Once fold dependence is accounted for, the evidence is
weak (p = {{cmp.hgb_vs_logreg_pr_auc.p_corrected:.2f}}). Boosting was still chosen as the primary model for its
better mean, lower variance, better calibration and exact SHAP explanations. The report does not claim it is
significantly better than the linear baseline. The gain over the label-free PCA model *does* survive the correction:
knowing what anomalies look like helps.

![Paired folds](figures/20_cv_paired_folds.png)

*Figure 6. Per-fold PR-AUC of gradient boosting against logistic regression on identical folds; points above the diagonal are folds where boosting wins.*

### 5.3 Calibration

The dashboard displays probabilities, so they should mean what they say. Both supervised models are close to the
diagonal (Figure 7). Gradient boosting has the lower Brier score ({{cv.hgb.cv_oof_at_threshold.brier:.3f}} vs
{{cv.logreg.cv_oof_at_threshold.brier:.3f}}), and the unweighted fit avoids the systematic over-prediction that class
re-weighting introduces. Novelty scores are not probabilities. The dashboard shows the PCA score as a percentile
among normal training segments instead.

![Calibration](figures/19_cv_calibration_logreg_vs_hgb.png)

*Figure 7. Reliability diagram (out-of-fold, train split) for logistic regression and gradient boosting.*

### 5.4 Novelty track

**PCA components show a clear bias–variance curve** (Figure 8). With one component the model cannot describe normal
behaviour, so normal segments reconstruct badly too (PR-AUC 0.71). With ten, the subspace is rich enough to
reconstruct anomalies as well (0.72). Three components are best (0.85), and the one-SE rule also selects them.

![PCA components](figures/21_pca_components.png)

*Figure 8. PCA reconstruction: grouped-CV PR-AUC against the number of components (mean ± sd over 15 folds).*

**The tracks are complementary.** PCA reconstruction flags **17 of the 56 anomalous segments** that boosting misses
in CV. On boosting's weakest channel, CADC0892, it raises recall from 0.44 to 0.63, at the cost of 28 false alarms
instead of 5 (Figure 9). This is the evidence for showing PCA as a *second opinion* in the review queue, not for
replacing the primary model.

**Isolation Forest fails in an instructive way.** It flags *every* normal CADC0892 segment (131) and 79 segments on
CADC0884, a channel with no anomalies at all. Without channel information, a global novelty model reads "this
channel's normal operating range is unusual" as "anomalous". A per-channel or channel-normalised novelty model is the
natural fix (Section 8).

![Per-channel recall and false alarms](figures/22_cv_channel_recall.png)

*Figure 9. Recall (left) and false alarms (right) per channel in CV, for each model at its own policy threshold.*

### 5.5 Explanations and error analysis

Global SHAP importance (Figure 10) is dominated by `n_peaks` and by `diff2_var` / `diff2_peaks`, the variance and
peak count of the second difference. These summarise spikes and abrupt changes in level. Channel identity contributes
little, which agrees with the weak channel-only baseline.

![SHAP global](figures/14_shap_global.png)

*Figure 10. Mean absolute out-of-sample SHAP value per feature (log-odds), gradient boosting, all training segments.*

![SHAP beeswarm](figures/15_shap_beeswarm.png)

*Figure 11. SHAP beeswarm: each dot is a held-out segment; colour is the feature value (red = high), position its effect on the log-odds of the anomalous label.*

**Where the model fails (CV).**

- **Misses** concentrate in CADC0872 (22) and CADC0892 (15; recall 0.44). They are typically segments with a single
  peak: the anomaly is a single event that the summary features dilute.
- **False alarms** concentrate in CADC0894 (26 false alarms against 13 true positives). These are long *normal*
  segments with many peaks, which look like the anomalous pattern on the dominant features.

Figures 12 and 13 show one confident detection and the most confidently missed anomaly.

![Waterfall TP](figures/16_shap_waterfall_tp.png)

*Figure 12. SHAP waterfall for a confidently detected anomalous segment (held-out). Peak-count features push the score up.*

![Waterfall FN](figures/17_shap_waterfall_fn.png)

*Figure 13. SHAP waterfall for the most confidently missed anomalous segment. Low peak counts push the score down; the anomaly is not visible in the summary features the model uses.*

### 5.6 Frozen test (scored once)

| Model | Test PR-AUC | 95% CI (episode bootstrap) | ROC-AUC | Precision | Recall | F1 | MCC | Alerts |
|---|---|---|---|---|---|---|---|---|
| Prevalence | {{test.prevalence.overall.pr_auc:.3f}} | – | {{test.prevalence.overall.roc_auc:.3f}} | {{test.prevalence.overall.precision:.2f}} | {{test.prevalence.overall.recall:.2f}} | {{test.prevalence.overall.f1:.2f}} | {{test.prevalence.overall.mcc:.2f}} | {{test.prevalence.overall.alerts}} |
| Channel only | {{test.channel_only.overall.pr_auc:.3f}} | [{{test.channel_only.bootstrap.pr_auc_ci95.0:.3f}}, {{test.channel_only.bootstrap.pr_auc_ci95.1:.3f}}] | {{test.channel_only.overall.roc_auc:.3f}} | {{test.channel_only.overall.precision:.2f}} | {{test.channel_only.overall.recall:.2f}} | {{test.channel_only.overall.f1:.2f}} | {{test.channel_only.overall.mcc:.2f}} | {{test.channel_only.overall.alerts}} |
| Logistic regression | {{test.logreg.overall.pr_auc:.3f}} | [{{test.logreg.bootstrap.pr_auc_ci95.0:.3f}}, {{test.logreg.bootstrap.pr_auc_ci95.1:.3f}}] | {{test.logreg.overall.roc_auc:.3f}} | {{test.logreg.overall.precision:.2f}} | {{test.logreg.overall.recall:.2f}} | {{test.logreg.overall.f1:.2f}} | {{test.logreg.overall.mcc:.2f}} | {{test.logreg.overall.alerts}} |
| **Gradient boosting** | **{{test.hgb.overall.pr_auc:.3f}}** | **[{{test.hgb.bootstrap.pr_auc_ci95.0:.3f}}, {{test.hgb.bootstrap.pr_auc_ci95.1:.3f}}]** | {{test.hgb.overall.roc_auc:.3f}} | **{{test.hgb.overall.precision:.2f}}** | **{{test.hgb.overall.recall:.2f}}** | {{test.hgb.overall.f1:.2f}} | **{{test.hgb.overall.mcc:.2f}}** | {{test.hgb.overall.alerts}} |
| PCA reconstruction | {{test.pca_recon.overall.pr_auc:.3f}} | [{{test.pca_recon.bootstrap.pr_auc_ci95.0:.3f}}, {{test.pca_recon.bootstrap.pr_auc_ci95.1:.3f}}] | {{test.pca_recon.overall.roc_auc:.3f}} | {{test.pca_recon.overall.precision:.2f}} | {{test.pca_recon.overall.recall:.2f}} | {{test.pca_recon.overall.f1:.2f}} | {{test.pca_recon.overall.mcc:.2f}} | {{test.pca_recon.overall.alerts}} |
| Isolation Forest | {{test.iforest.overall.pr_auc:.3f}} | [{{test.iforest.bootstrap.pr_auc_ci95.0:.3f}}, {{test.iforest.bootstrap.pr_auc_ci95.1:.3f}}] | {{test.iforest.overall.roc_auc:.3f}} | {{test.iforest.overall.precision:.2f}} | {{test.iforest.overall.recall:.2f}} | {{test.iforest.overall.f1:.2f}} | {{test.iforest.overall.mcc:.2f}} | {{test.iforest.overall.alerts}} |

*Table 3. The frozen test split (529 segments, {{test.hgb.overall.prevalence:.1%}} anomalous), scored once at thresholds fixed from training OOF scores.*

On 529 unseen segments the primary model reaches **PR-AUC {{test.hgb.overall.pr_auc:.3f}}** (95% CI
{{test.hgb.bootstrap.pr_auc_ci95.0:.3f}}–{{test.hgb.bootstrap.pr_auc_ci95.1:.3f}}). At the pre-registered threshold
it flags {{test.hgb.overall.alerts}} segments. Of these, 97 carry the anomalous label (precision
{{test.hgb.overall.precision:.2f}}); 16 of 113 anomalous segments are missed (recall {{test.hgb.overall.recall:.2f}}).
The model ranking is identical to CV. The probabilities remain well calibrated (Brier
{{test.hgb.overall.brier:.3f}}).

![Test PR and confusion](figures/24_test_pr_confusion.png)

*Figure 14. Frozen test: precision–recall curves for all models (left) and the primary model's confusion matrix at the pre-registered threshold (right).*

**Why test is higher than CV.** Every model scores higher on test than in grouped CV. The primary model's test score
({{test.hgb.overall.pr_auc:.3f}}) is almost exactly the *random*-CV estimate from Section 3
({{leak.random.model_pr_auc.mean:.3f}}), as predicted. 89% of test segments neighbour a training segment, and the
final model was fit on all of those neighbours. **Grouped CV ({{cv.hgb.cv_folds.pr_auc_mean:.3f}}) remains the
estimate we would quote for new, unseen periods of telemetry.** The test figure is reported for comparability with
the benchmark.

![CV vs test](figures/23_cv_vs_test.png)

*Figure 15. Grouped-CV estimate (mean ± sd) against frozen-test PR-AUC (95% episode-bootstrap CI) per model; the cross marks the leaky random-CV estimate from Section 3.*

**Isolated test segments.** The {{test.hgb.isolated_segments.n}} test segments that touch no training segment give
PR-AUC {{test.hgb.isolated_segments.pr_auc:.3f}} for boosting and {{test.logreg.isolated_segments.pr_auc:.3f}} for
logistic regression. The subset is small and unrepresentative ({{test.hgb.isolated_segments.prevalence:.0%}}
anomalous, against {{test.hgb.overall.prevalence:.0%}} overall), so it shows that performance does not collapse away
from neighbours, but nothing stronger.

**Per channel (primary model, test).**

| Channel | n | Anomaly rate | PR-AUC | Precision | Recall | Alerts |
|---|---|---|---|---|---|---|
| CADC0872 | {{test.hgb.per_channel.CADC0872.n}} | {{test.hgb.per_channel.CADC0872.prevalence:.2f}} | {{test.hgb.per_channel.CADC0872.pr_auc:.3f}} | {{test.hgb.per_channel.CADC0872.precision:.2f}} | {{test.hgb.per_channel.CADC0872.recall:.2f}} | {{test.hgb.per_channel.CADC0872.alerts}} |
| CADC0873 | {{test.hgb.per_channel.CADC0873.n}} | {{test.hgb.per_channel.CADC0873.prevalence:.2f}} | {{test.hgb.per_channel.CADC0873.pr_auc:.3f}} | {{test.hgb.per_channel.CADC0873.precision:.2f}} | {{test.hgb.per_channel.CADC0873.recall:.2f}} | {{test.hgb.per_channel.CADC0873.alerts}} |
| CADC0874 | {{test.hgb.per_channel.CADC0874.n}} | {{test.hgb.per_channel.CADC0874.prevalence:.2f}} | {{test.hgb.per_channel.CADC0874.pr_auc:.3f}} | {{test.hgb.per_channel.CADC0874.precision:.2f}} | {{test.hgb.per_channel.CADC0874.recall:.2f}} | {{test.hgb.per_channel.CADC0874.alerts}} |
| CADC0888 | {{test.hgb.per_channel.CADC0888.n}} | {{test.hgb.per_channel.CADC0888.prevalence:.2f}} | {{test.hgb.per_channel.CADC0888.pr_auc:.3f}} | {{test.hgb.per_channel.CADC0888.precision:.2f}} | {{test.hgb.per_channel.CADC0888.recall:.2f}} | {{test.hgb.per_channel.CADC0888.alerts}} |
| CADC0892 | {{test.hgb.per_channel.CADC0892.n}} | {{test.hgb.per_channel.CADC0892.prevalence:.2f}} | {{test.hgb.per_channel.CADC0892.pr_auc:.3f}} | {{test.hgb.per_channel.CADC0892.precision:.2f}} | {{test.hgb.per_channel.CADC0892.recall:.2f}} | {{test.hgb.per_channel.CADC0892.alerts}} |
| **CADC0894** | {{test.hgb.per_channel.CADC0894.n}} | {{test.hgb.per_channel.CADC0894.prevalence:.2f}} | **{{test.hgb.per_channel.CADC0894.pr_auc:.3f}}** | **{{test.hgb.per_channel.CADC0894.precision:.2f}}** | {{test.hgb.per_channel.CADC0894.recall:.2f}} | {{test.hgb.per_channel.CADC0894.alerts}} |
| CADC0886 | {{test.hgb.per_channel.CADC0886.n}} | {{test.hgb.per_channel.CADC0886.prevalence:.2f}} | {{test.hgb.per_channel.CADC0886.pr_auc:.3f}} | {{test.hgb.per_channel.CADC0886.precision:.2f}} | {{test.hgb.per_channel.CADC0886.recall:.2f}} | {{test.hgb.per_channel.CADC0886.alerts}} |
| CADC0884 | {{test.hgb.per_channel.CADC0884.n}} | 0 anomalous | – | – | – | {{test.hgb.per_channel.CADC0884.flagged}} |
| CADC0890 | {{test.hgb.per_channel.CADC0890.n}} | all anomalous | – | – | {{test.hgb.per_channel.CADC0890.flagged}} / {{test.hgb.per_channel.CADC0890.anomalies}} flagged | {{test.hgb.per_channel.CADC0890.flagged}} |

*Table 4. Primary model on the frozen test, by channel. CADC0884 and CADC0890 have one class in the test split, so no ranking metric exists; CADC0886 has four segments and is anecdotal.*

**The CV weak spot replicates on test.** CADC0894 has the lowest PR-AUC and half of its alerts are false alarms, the
same failure mode as in CV (long multi-peak normal segments). The independent PCA score does not fix this channel. A
reviewer should treat CADC0894 alerts with extra caution, and the dashboard's per-channel table makes this visible.

<div class="pagebreak"></div>

## 6. Dashboard

### 6.1 Architecture and artifact provenance

The dashboard is a five-page Streamlit application (`app/`). It is deliberately **read-only with respect to models**:
it never fits or unpickles a model, so what it shows cannot drift from what was evaluated, and it has no runtime
coupling to library versions. A build step (`python -m satelliteguard.dashboard.extract`) writes four small committed
files (2.7 MB in total), and the app reads only these files and the JSON reports.

| Artifact | Source | Contents |
|---|---|---|
| `dashboard/segments.csv` | OOF predictions (train), `test_predictions.csv` (test) | metadata, scores and flags of boosting, logistic and PCA; novelty percentile; score provenance |
| `dashboard/shap.csv` | held-out fold models (train), final model (test) | per-segment SHAP values, base value, log-odds |
| `dashboard/features.csv` | publisher feature table | feature values for comparison with normal segments |
| `dashboard/traces.csv.gz` | raw `segments.csv` | samples per segment (seconds since start, value) |
| `reports/*.json` | evaluation pipeline | all metrics, comparisons, leakage check, test lock |

Contract tests check that the extract covers every segment exactly once, and that the test flags equal the frozen-test
predictions. They also check that training scores equal the OOF scores, that SHAP values add up to the model's
log-odds, and that test-segment SHAP reproduces the final model's probabilities. A headless test then runs every page
and fails on any exception. Because the app needs no raw download, it can be deployed directly to Streamlit Community
Cloud.

**Safety language.** Every page shows a scope note: a flag marks a segment for review and is not a fault diagnosis.
Labels are called *benchmark labels*. Threshold controls default to the pre-registered value, and any other value
triggers an "exploratory" warning, so a re-tuned number cannot be mistaken for a reported result.

### 6.2 User flows

1. **Triage.** Open *Review queue* (test split, policy threshold), read the ranked list with the boosting probability,
   novelty percentile and agreement column, and export the queue as CSV for offline review.
2. **Investigate one segment.** Select a row, then *Open in explorer*. Inspect the raw trace in the context of its
   episode neighbours, read *why the model scored it this way* (top SHAP contributions), and compare the segment's
   feature values with normal segments on the same channel.
3. **Decide how much to trust the model.** In *Model comparison*, look at the precision–recall curves, move the
   threshold to see the alert/miss trade-off, check the per-channel table (CADC0894 caveat) and the significance
   tests.
4. **Audit the data.** *Data health* shows the contract checks, class balance by channel, the leakage summary and
   provenance hashes.

<figure class="tall"><img src="figures/dashboard_overview.png" alt="Dashboard overview"><figcaption>Figure 16. Overview page: scope statement, headline frozen-test metrics with the bootstrap interval, all-model table, and the explanation of why test scores exceed grouped CV.</figcaption></figure>

<figure class="tall"><img src="figures/dashboard_review_queue.png" alt="Dashboard review queue"><figcaption>Figure 17. Review queue (test split, pre-registered threshold): 106 of 529 segments queued, ranked by boosting probability, with the PCA novelty percentile as an independent second opinion and an agreement column. Queue settings are in the sidebar; benchmark labels can be hidden to see the queue as an operator would. Export to CSV at the bottom.</figcaption></figure>

<figure class="tall"><img src="figures/dashboard_segment_explorer_part1.png" alt="Segment explorer, scores and trace"><figcaption>Figure 18a. Segment explorer for the top-ranked test segment. Scores from the three models and the benchmark label, then the raw trace (highlighted) with two neighbouring segments on each side of the same episode, coloured by benchmark label.</figcaption></figure>

<figure class="tall"><img src="figures/dashboard_segment_explorer_part2.png" alt="Segment explorer, explanation"><figcaption>Figure 18b. "Why did the model score it this way?": SHAP contributions in log-odds (from 0.08 for an average segment to 1.00 for this one), the features pushing in each direction, and the segment's feature values as percentiles among normal training segments on the same channel.</figcaption></figure>

<figure class="tall"><img src="figures/dashboard_model_comparison_part1.png" alt="Model comparison, curves and threshold"><figcaption>Figure 19a. Model comparison: interactive precision–recall curves on the frozen test with the policy operating points (crosses), and the exploratory threshold trade-off with a live confusion matrix. At the pre-registered threshold the numbers match Table 3.</figcaption></figure>

<figure class="tall"><img src="figures/dashboard_model_comparison_part2.png" alt="Model comparison, channels and tests"><figcaption>Figure 19b. Model comparison, continued: per-channel test results, the corrected significance tests and global SHAP importance.</figcaption></figure>

<figure class="tall"><img src="figures/dashboard_data_health_part1.png" alt="Data health, contract and balance"><figcaption>Figure 20a. Data health: the twelve data-contract checks and class balance by split and channel.</figcaption></figure>

<figure class="tall"><img src="figures/dashboard_data_health_part2.png" alt="Data health, leakage and provenance"><figcaption>Figure 20b. Data health, continued: the neighbour-leakage summary (random vs grouped CV, episode timeline) and provenance.</figcaption></figure>

<div class="pagebreak"></div>

## 7. Reproducibility and engineering

| Practice | Implementation |
|---|---|
| Data integrity | checksum-verified download; `data/manifest.json`; 12-check data contract gates the pipeline |
| Single source of truth | all metrics in `reports/metrics.json` (+ comparisons, leakage check, test lock); this report and the dashboard are generated from them |
| Leakage controls | transforms inside pipelines; episode-grouped folds; early stopping disabled; tests assert no episode crosses a fold |
| Test-set discipline | pre-registration in the decision log; run-once lock file with commit hash; test asserts the lock refuses a second run |
| Environment | `requirements-lock.txt` pins the exact versions used (Python 3.13, scikit-learn 1.9.1) |
| Automation | 58 pytest tests (data contract, metrics, grouped CV, one-SE rule, corrected t-test, SHAP additivity, novelty models, test-prediction contract, dashboard, report and README/model-card numbers vs `metrics.json`); ruff lint; GitHub Actions CI (Ubuntu, Python 3.11) green on every commit |
| Traceability | `docs/decision_log.md` records each day's decisions and evidence; `docs/model_card.md` |

**Clean-environment dry run.** `scripts/reproduce.py` re-runs the whole pipeline (download, validation, episodes,
leakage check, tuning of all four families, training, comparisons, SHAP, frozen-test evaluation, figures, dashboard
extract and the test suite). It then compares every value in `metrics.json`, `comparisons.json` and
`leakage_check.json` with the committed versions. The script was run on a fresh clone in a new virtual environment
built only from `requirements-lock.txt` (Windows, Python 3.13; 53 minutes end to end). **All 1,077 reported values
were reproduced exactly** (992 in `metrics.json`, 66 in `comparisons.json`, 19 in `leakage_check.json`; tolerance
10⁻⁶; timestamps and commit ids excluded). Per-segment outputs (OOF scores, SHAP values, test predictions, dashboard
extracts) were byte-identical, and the test suite passed. On the network used, the Zenodo *file* endpoint failed TLS
verification, so the raw CSVs were copied into the clone. The download step still verified them against the
checksums Zenodo publishes before anything else ran. In that throwaway clone, the frozen test is re-scored with
`--force` purely to verify the numbers. The committed lock file remains the record of the single official run.

The report itself is built with `python -m satelliteguard.report`. Every number in it is a placeholder resolved from
the JSON reports, and an unresolved placeholder is a build error. The PDF is rendered from the same Markdown
(`reports/technical_report.md`) by headless Chrome. Dashboard screenshots are regenerated with
`scripts/capture_dashboard.py`.

## 8. Limitations and next steps

**What the results do and do not establish.**

- **Benchmark scope.** This is a public-benchmark prototype, not a flight-qualified system. Results cover nine channels
  of one mission at two sampling rates. They do not establish generalisation to other missions, unseen channels or
  other data-rate regimes.
- **Label semantics.** Segment labels are benchmark annotations that may reflect context the model never sees. They are
  not confirmed hardware faults, and label noise propagates directly into every metric.
- **Generalisation in time.** The publisher split is interleaved, and no temporal hold-out exists. Grouped CV is the
  best available estimate for new episodes, but drift over the mission lifetime is untested.
- **Small channels.** CADC0886 and CADC0890 (11 and 14 segments) are effectively unvalidated, and CADC0884 and CADC0890
  have a single class in the test split.
- **Explanations are associations.** SHAP describes how the model uses the publisher features, not physical
  causation.
- **Features are fixed.** The model consumes publisher summary features, and single-event anomalies that these
  features dilute are the main source of misses.
- **Statistical power.** With 15 correlated folds, differences of about 0.04 PR-AUC between strong models cannot be
  resolved. The corrected test says so explicitly.

**Next steps.**

1. **Per-channel or channel-normalised novelty models**, to remove Isolation Forest's channel-scale confusion and
   possibly give a stronger second opinion.
2. **A temporal split** (train on earlier episodes, test on later ones) as a more realistic estimate of future
   performance.
3. **Sequence models on raw samples** (1-D convolutional or recurrent), the proposal's stretch track, aimed at the
   single-peak anomalies the summary features miss.
4. **Alert-cost design** with domain experts: replace "recall ≥ 0.80" by explicit costs of misses and false alarms,
   and per-channel thresholds where justified (for example CADC0894).
5. **Operational validation before any real use:** domain review of flagged segments, temporal back-testing,
   monitoring for drift and calibration decay, governance and access control, and human-in-the-loop evaluation of the
   dashboard.

## 9. Conclusion

On OPSSAT-AD the dominant source of optimistic results is not model choice but **how the data are split**. Contiguous
segments leak information across random folds and across the publisher's interleaved split, inflating PR-AUC by about
{{leak.optimism_pr_auc:.2f}}. With episode-grouped validation, bounded tuning, a pre-declared threshold and a
once-only frozen test, a gradient-boosting model ranks anomalous segments well:

- **Grouped CV (honest estimate for new periods):** PR-AUC {{cv.hgb.cv_folds.pr_auc_mean:.2f}}.
- **Benchmark test (interleaved split):** PR-AUC {{test.hgb.overall.pr_auc:.2f}}, precision
  {{test.hgb.overall.precision:.2f}} and recall {{test.hgb.overall.recall:.2f}}, flagging about one segment in five.

A label-free PCA model is a useful second opinion, and its gap to the supervised model is real. The dashboard turns
these results into a review workflow that shows its evidence and its limits. It ranks segments for human review and
makes no claim to diagnose faults.

## 10. References

1. Ruszczak, B., Kotowski, K., Evans, D., Nalepa, J. *OPSSAT-AD — anomaly detection dataset for satellite telemetry*
   (v2). Zenodo, record 15108715. https://zenodo.org/records/15108715
2. Ruszczak, B., Kotowski, K., Evans, D., Nalepa, J. The OPS-SAT benchmark for detecting anomalies in satellite
   telemetry. *Scientific Data* (2025). https://doi.org/10.1038/s41597-025-05035-3
3. OPS-SAT-AD source repository and notebooks. https://github.com/kplabs-pl/OPS-SAT-AD
4. Ruszczak, B. et al. Machine learning detects anomalies in OPS-SAT telemetry. *ICCS 2023*, LNCS.
   https://doi.org/10.1007/978-3-031-35995-8_21
5. Nadeau, C., Bengio, Y. Inference for the generalization error. *Machine Learning* 52, 239–281 (2003).
6. Lundberg, S. M. et al. From local explanations to global understanding with explainable AI for trees. *Nature
   Machine Intelligence* 2, 56–67 (2020).
7. Liu, F. T., Ting, K. M., Zhou, Z.-H. Isolation Forest. *ICDM* (2008).
8. Ke, G. et al. LightGBM: a highly efficient gradient boosting decision tree. *NeurIPS* (2017). (Histogram-based
   boosting, as implemented in scikit-learn's `HistGradientBoostingClassifier`.)
9. Saito, T., Rehmsmeier, M. The precision-recall plot is more informative than the ROC plot when evaluating binary
   classifiers on imbalanced datasets. *PLoS ONE* 10(3) (2015).
10. Mitchell, M. et al. Model cards for model reporting. *FAT\** (2019).
11. Pedregosa, F. et al. Scikit-learn: machine learning in Python. *JMLR* 12, 2825–2830 (2011).

<div class="pagebreak"></div>

## 11. Appendices

### A. Hyperparameter grids (episode-grouped CV, 15 folds)

The selected configuration is the simplest one within one standard error of the best (bold).

**Logistic regression** — C ∈ {0.01, 0.1, 1, 10, 100} × scaling ∈ {quantile, standard}. Best: C = 100,
quantile (0.889 ± 0.066). **Selected: C = 1, quantile ({{cv.logreg.cv_folds.pr_auc_mean:.3f}} ±
{{cv.logreg.cv_folds.pr_auc_std:.3f}}).**

| max leaf nodes \\ learning rate | 0.03 | 0.1 |
|---|---|---|
| 4 | **0.919 ± 0.060** | 0.925 ± 0.056 |
| 8 | 0.925 ± 0.063 | 0.929 ± 0.052 |
| 16 | 0.926 ± 0.057 | 0.930 ± 0.053 |
| 31 | 0.926 ± 0.055 | 0.930 ± 0.052 |

*Gradient boosting (300 trees, min 20 samples per leaf, L2 = 1). All configurations lie within one standard error of the best; the simplest is selected.*

| PCA components | 1 | 2 | **3** | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|
| PR-AUC | 0.715 | 0.819 | **0.850** | 0.790 | 0.775 | 0.744 | 0.717 |

| Isolation Forest max_samples \\ max_features | 0.5 | 1.0 |
|---|---|---|
| 128 | 0.590 ± 0.145 | **0.611 ± 0.160** |
| 256 | 0.614 ± 0.167 | 0.638 ± 0.162 |
| 512 | 0.611 ± 0.172 | 0.648 ± 0.178 |

### B. Univariate feature screen (train split)

Single-feature ROC-AUC, |AUC − 0.5| × 2 as strength, and train-vs-test Kolmogorov–Smirnov statistic (top features).

| Feature | ROC-AUC | Strength | KS (train vs test) | KS p-value |
|---|---|---|---|---|
| n_peaks | 0.836 | 0.672 | 0.018 | 0.999 |
| kurtosis | 0.716 | 0.431 | 0.040 | 0.521 |
| smooth10_n_peaks | 0.701 | 0.401 | 0.017 | 1.000 |
| len | 0.691 | 0.383 | 0.039 | 0.566 |

### C. Metric definitions

- **PR-AUC (average precision):** AP = Σₖ (Rₖ − Rₖ₋₁) Pₖ over score thresholds. Chance level equals the positive
  rate.
- **Recall / precision @ top 25%:** recall and precision among the 25% highest-scored segments (a fixed review
  budget).
- **MCC:** (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)), balanced even under class imbalance.
- **Brier score:** mean of (p − y)², for probabilistic models only.
- **Episode bootstrap:** resample whole test episodes with replacement (2,000 times), recompute the metric at the
  fixed threshold, and report the 2.5th and 97.5th percentiles.

### D. Repository map

```
configs/base.yaml            seed, paths, CV settings
src/satelliteguard/data/     download (checksums), validate (12-check contract), episodes
src/satelliteguard/          features.py (pipelines), eda.py, report.py
src/satelliteguard/models/   registry (model matrix), tune (grids + one-SE), train, explain (SHAP)
src/satelliteguard/evaluation/  metrics, grouped CV, compare (corrected t-test), leakage_check,
                             evaluate_test (run-once lock + episode bootstrap), plots
src/satelliteguard/dashboard/   extract (committed artifacts), queries (tested view logic)
app/                         Streamlit dashboard (5 pages)
notebooks/01-04              executed analysis notebooks
reports/                     metrics.json, comparisons.json, test_lock.json, figures/, this report
scripts/                     reproduce.py (clean-run diff), capture_dashboard.py
docs/                        decision_log.md, model_card.md, report sources
tests/                       58 tests; CI in .github/workflows/ci.yml
```
