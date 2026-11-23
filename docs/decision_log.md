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

**Next (D1)**
- Review data audit, EDA notebook, raw-trace gallery.
- Check temporal adjacency of segments across CV folds (neighbour-leakage risk).
