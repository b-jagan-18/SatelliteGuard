# Changelog

## v1.0.0 — 25 Nov 2026 (course submission)

First complete release: leakage-safe benchmark, decision-support dashboard and technical report
for anomaly ranking on OPSSAT-AD (ESA OPS-SAT telemetry).

**Results** (primary model: histogram gradient boosting)
- Episode-grouped CV PR-AUC 0.919 ± 0.060 (generalisation estimate for unseen periods).
- Frozen test, scored once: PR-AUC 0.962 [0.941, 0.985], precision 0.92, recall 0.86 at the
  pre-registered threshold 0.389.
- Random CV overstates PR-AUC by 0.036 because of neighbour leakage between contiguous segments.

**Contents**
- `src/satelliteguard/`: checksum-verified download, 12-check data contract, episode grouping,
  grouped CV, bounded tuning with the one-SE rule, corrected paired t-tests, out-of-sample SHAP,
  run-once frozen-test evaluator with episode bootstrap, dashboard extract, report builder.
- `app/`: five-page Streamlit dashboard (overview, review queue with CSV export, segment explorer
  with SHAP, model comparison, data health) that runs on committed artifacts only.
- `reports/technical_report.pdf`: 32-page report generated from `reports/*.json`.
- `docs/model_card.md`, `docs/decision_log.md` (D0–D7).
- `notebooks/01–04`: executed analysis notebooks.
- 58 tests, ruff, GitHub Actions CI; `scripts/reproduce.py` + `requirements-lock.txt` (clean-clone
  re-run reproduced all 1,077 reported values exactly).

### Development history
- Day 0 — scaffold, download + manifest, data contract.
- Day 1 — EDA, contiguous episodes, neighbour-leakage experiment.
- Day 2 — metrics, grouped CV, threshold policy, logistic baseline.
- Day 3 — gradient boosting, corrected t-test, SHAP.
- Day 4 — novelty models (PCA reconstruction, Isolation Forest), pre-registration, frozen test.
- Day 5 — Streamlit dashboard.
- Day 6 — technical report, model card, reproducibility dry run.
- Day 7 — consistency tests for README/model card, final README, release.
