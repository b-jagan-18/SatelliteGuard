# SatelliteGuard

**Machine learning-based anomaly detection in satellite telemetry for early fault identification.**

SatelliteGuard is a reproducible, leakage-safe benchmark and decision-support dashboard built on
the ESA OPS-SAT telemetry dataset (OPSSAT-AD). It ranks telemetry segments for operator review.

> **Scope:** the system flags *anomalous segments* for review. It does **not** diagnose or
> confirm spacecraft hardware faults. Labels are segment-level benchmark annotations, and channel
> IDs (e.g. `CADC0872`) are opaque — no subsystem meaning is claimed.

**Status:** 🚧 in progress (deadline 30 Nov 2026) — see [docs/decision_log.md](docs/decision_log.md).

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python -m satelliteguard.data.download   # fetch + checksum-verify Zenodo files
python -m satelliteguard.data.validate   # data-contract audit (red gate)
pytest -q
```

## Data

| File | Contents |
|---|---|
| `segments.csv` | 303,493 raw telemetry samples grouped into segments |
| `dataset.csv` | 2,123 segments × engineered summary features |

| Split | Normal | Anomalous | Total |
|---|---|---|---|
| train (`train=1`) | 1,273 | 321 | 1,594 |
| frozen test (`train=0`) | 416 | 113 | 529 |

Source: Ruszczak et al., *OPSSAT-AD — anomaly detection dataset for satellite telemetry*,
Zenodo [15108715](https://zenodo.org/records/15108715), CC-BY-4.0.
File hashes are recorded in [data/manifest.json](data/manifest.json).

## Approach

- **Supervised:** calibrated logistic regression, gradient-boosted trees (class-weighted).
- **Unsupervised / novelty:** Isolation Forest and PCA reconstruction, fit on train-normal only.
- **Validation:** repeated stratified CV inside the train split; every transform is fit inside
  folds; thresholds chosen from CV predictions; the frozen test is scored once.
- **Primary metric:** PR-AUC (≈20% anomaly rate), plus recall/precision at a review budget,
  MCC, calibration.

## Repository layout

```
configs/            experiment config (paths, seed, CV)
data/raw/           Zenodo downloads (git-ignored, re-downloadable)
data/processed/     audit, feature tables, dashboard extracts
src/satelliteguard/ data, models, evaluation packages
app/                Streamlit dashboard
notebooks/          EDA and analysis
reports/            figures, metrics.json, technical report
tests/              data-contract and pipeline tests
docs/               decision log
```

## References

1. Ruszczak, B. et al. OPSSAT-AD — anomaly detection dataset for satellite telemetry. Zenodo 15108715 (v2).
2. Ruszczak, B., Kotowski, K., Evans, D., Nalepa, J. The OPS-SAT benchmark for detecting anomalies in
   satellite telemetry. *Scientific Data* (2025). https://doi.org/10.1038/s41597-025-05035-3
3. OPS-SAT-AD repository: https://github.com/kplabs-pl/OPS-SAT-AD
