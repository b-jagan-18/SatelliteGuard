"""Segment index and contiguous-episode grouping.

OPS-SAT segments are back-to-back slices of continuous telemetry: on a given channel the next
segment usually starts one sampling step after the previous one ends. Neighbouring segments are
therefore highly correlated (and anomalies cluster), so random CV folds leak information.

An *episode* is a maximal run of segments on one channel where each gap to the previous segment
is <= ``max_gap_s``. Episodes are used as groups for cross-validation.

Usage:
    python -m satelliteguard.data.episodes     # writes data/processed/segment_index.csv
"""

from __future__ import annotations

import pandas as pd

from satelliteguard import load_config

DEFAULT_MAX_GAP_S = 10.0


def build_segment_index(
    segments: pd.DataFrame, max_gap_s: float = DEFAULT_MAX_GAP_S
) -> pd.DataFrame:
    """One row per segment: channel, time span, gap to previous segment, episode id, neighbours."""
    ts = pd.to_datetime(segments["timestamp"], utc=True)
    idx = (
        segments.assign(timestamp=ts)
        .groupby("segment")
        .agg(
            channel=("channel", "first"),
            start=("timestamp", "min"),
            end=("timestamp", "max"),
            n_samples=("value", "size"),
            sampling=("sampling", "first"),
            anomaly=("anomaly", "first"),
            train=("train", "first"),
        )
        .reset_index()
        .sort_values(["channel", "start"])
        .reset_index(drop=True)
    )
    by_ch = idx.groupby("channel")
    idx["gap_prev_s"] = (idx["start"] - by_ch["end"].shift()).dt.total_seconds()
    new_episode = idx["gap_prev_s"].isna() | (idx["gap_prev_s"] > max_gap_s)
    idx["episode"] = new_episode.cumsum().astype(int) - 1
    idx["pos_in_episode"] = idx.groupby("episode").cumcount()

    # Is the segment directly touching a segment from the other split?
    gap_next = by_ch["gap_prev_s"].shift(-1)
    prev_other = (by_ch["train"].shift() != idx["train"]) & (idx["gap_prev_s"] <= max_gap_s)
    next_other = (by_ch["train"].shift(-1) != idx["train"]) & (gap_next <= max_gap_s)
    idx["touches_other_split"] = (prev_other | next_other).fillna(False)
    return idx


def leakage_summary(idx: pd.DataFrame) -> dict:
    """Headline numbers on neighbour dependence, for the report and decision log."""
    prev_anom = idx.groupby("channel")["anomaly"].shift()
    ep = idx.groupby("episode")
    return {
        "n_episodes": int(idx["episode"].nunique()),
        "episode_size_median": float(ep.size().median()),
        "episode_size_max": int(ep.size().max()),
        "episodes_spanning_both_splits": int(ep["train"].nunique().gt(1).sum()),
        "test_segments_touching_train": int(
            (idx["touches_other_split"] & idx["train"].eq(0)).sum()
        ),
        "n_test_segments": int(idx["train"].eq(0).sum()),
        "p_anomaly_given_prev_anomaly": round(float(idx.loc[prev_anom == 1, "anomaly"].mean()), 3),
        "p_anomaly_given_prev_normal": round(float(idx.loc[prev_anom == 0, "anomaly"].mean()), 3),
    }


def main() -> None:
    cfg = load_config()
    segments = pd.read_csv(cfg["paths"]["raw"] / "segments.csv")
    idx = build_segment_index(segments)
    out = cfg["paths"]["processed"] / "segment_index.csv"
    idx.to_csv(out, index=False)
    print(f"Wrote {out} ({len(idx)} segments, {idx['episode'].nunique()} episodes)")
    for k, v in leakage_summary(idx).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
