"""Day-1 data contract: verify schema, split integrity and label consistency.

Writes data/processed/data_audit.json. Any failed check is a red gate:
no model results are reportable until it is resolved.

Usage:
    python -m satelliteguard.data.validate
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from satelliteguard import load_config

SEGMENTS_COLS = [
    "channel", "timestamp", "value", "label", "sampling", "anomaly", "segment", "train",
]
EXPECTED_SPLIT = {  # (train flag, anomaly) -> segment count, from the publisher record
    (1, 0): 1273,
    (1, 1): 321,
    (0, 0): 416,
    (0, 1): 113,
}


def load_raw(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = cfg["paths"]["raw"]
    segments = pd.read_csv(raw / "segments.csv")
    dataset = pd.read_csv(raw / "dataset.csv")
    return segments, dataset


def run_checks(segments: pd.DataFrame, dataset: pd.DataFrame) -> dict:
    checks: dict[str, dict] = {}

    def record(name: str, passed: bool, detail=None):
        checks[name] = {"passed": bool(passed), "detail": detail}

    # --- schema -----------------------------------------------------------
    record(
        "segments_columns",
        list(segments.columns) == SEGMENTS_COLS,
        list(segments.columns),
    )
    required = {"segment", "anomaly", "train", "channel", "sampling"}
    record("dataset_required_columns", required <= set(dataset.columns), list(dataset.columns))

    # --- uniqueness / nulls ------------------------------------------------
    record("dataset_unique_segment_ids", dataset["segment"].is_unique,
           int(dataset["segment"].duplicated().sum()))
    record("dataset_no_nulls", int(dataset.isna().sum().sum()) == 0,
           {k: int(v) for k, v in dataset.isna().sum().items() if v})
    record("segments_no_nulls", int(segments.isna().sum().sum()) == 0,
           {k: int(v) for k, v in segments.isna().sum().items() if v})

    # --- one channel / target / split per segment ---------------------------
    per_seg = segments.groupby("segment")[["channel", "anomaly", "train", "sampling"]].nunique()
    bad = per_seg[(per_seg > 1).any(axis=1)]
    record("one_channel_target_split_per_segment", bad.empty, bad.index.tolist()[:20])

    # --- raw 'label' vs numeric 'anomaly' -----------------------------------
    xtab = pd.crosstab(segments["label"], segments["anomaly"])
    record("label_vs_anomaly_crosstab", True, xtab.to_dict())

    # --- segment sets agree between files -----------------------------------
    seg_ids, ds_ids = set(segments["segment"]), set(dataset["segment"])
    record("segment_ids_match_between_files", seg_ids == ds_ids,
           {"only_in_segments": len(seg_ids - ds_ids), "only_in_dataset": len(ds_ids - seg_ids)})

    # --- target/split agree between files -----------------------------------
    seg_meta = segments.groupby("segment")[["anomaly", "train", "channel"]].first()
    cols = ["anomaly", "train", "channel"]
    merged = dataset.set_index("segment")[cols].merge(
        seg_meta, left_index=True, right_index=True, suffixes=("", "_raw"))
    mismatch = {c: int((merged[c] != merged[f"{c}_raw"]).sum()) for c in cols}
    record("target_split_channel_agree_between_files", sum(mismatch.values()) == 0, mismatch)

    # --- publisher split counts ---------------------------------------------
    counts = dataset.groupby(["train", "anomaly"]).size().to_dict()
    counts = {f"train={k[0]},anomaly={k[1]}": int(v) for k, v in counts.items()}
    expected = {f"train={k[0]},anomaly={k[1]}": v for k, v in EXPECTED_SPLIT.items()}
    record("split_counts_match_publisher", counts == expected, counts)

    # --- timestamps ---------------------------------------------------------
    ts = segments.assign(timestamp=pd.to_datetime(segments["timestamp"], utc=True, errors="coerce"))
    record("timestamps_parse", ts["timestamp"].notna().all(), int(ts["timestamp"].isna().sum()))
    non_mono = [s for s, g in ts.groupby("segment")["timestamp"] if not g.is_monotonic_increasing]
    record("timestamps_monotonic_within_segment", not non_mono,
           {"n_segments": len(non_mono), "examples": non_mono[:10]})

    # --- informational ------------------------------------------------------
    info = {
        "n_raw_rows": int(len(segments)),
        "n_segments": int(dataset["segment"].nunique()),
        "channels": dataset["channel"].value_counts().sort_index().to_dict(),
        "anomaly_rate_by_channel": dataset.groupby("channel")["anomaly"].mean().round(3).to_dict(),
        "segment_length": ts.groupby("segment").size().describe().round(1).to_dict(),
    }
    return {"all_passed": all(c["passed"] for c in checks.values()), "checks": checks, "info": info}


def main() -> int:
    cfg = load_config()
    segments, dataset = load_raw(cfg)
    audit = run_checks(segments, dataset)
    out = cfg["paths"]["audit"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")

    for name, c in audit["checks"].items():
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {name}")
    print(f"\nAudit written to {out}")
    if not audit["all_passed"]:
        print("RED GATE: resolve failed checks before reporting any model results.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
