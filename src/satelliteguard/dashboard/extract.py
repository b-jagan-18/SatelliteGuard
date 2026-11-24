"""Build the dashboard extracts in data/processed/dashboard/.

The dashboard never fits or loads a model. It reads these small files plus reports/*.json, so it
runs without the raw Zenodo download (e.g. on Streamlit Community Cloud).

Score provenance (shown in the app):
- train segments: out-of-fold scores from episode-grouped CV (mean over 3 repeats), i.e. each
  score comes from models that never saw that segment's episode;
- test segments: final models fit on the full train split, scored once (test_predictions.csv).
SHAP for train segments comes from held-out fold models (first CV repeat, explain.py); for test
segments it is computed here with the final gradient-boosting model.

Usage:
    python -m satelliteguard.dashboard.extract
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from satelliteguard import PROJECT_ROOT, load_config
from satelliteguard.features import load_modeling_table, numeric_features, split_xy
from satelliteguard.models.explain import shap_frame

SCORED = ["hgb", "logreg", "pca_recon"]
SHAP_META = ["segment", "channel_id", "anomaly", "logit", "base_value"]


def segment_table(cfg: dict) -> pd.DataFrame:
    """One row per segment: metadata, scores/flags of the dashboard models, novelty percentile."""
    proc = cfg["paths"]["processed"]
    metrics = pd.read_json(cfg["paths"]["metrics"], typ="series")
    idx = pd.read_csv(proc / "segment_index.csv")
    seg = idx[["segment", "channel", "start", "end", "n_samples", "sampling", "anomaly", "train",
               "episode", "pos_in_episode", "touches_other_split"]].copy()
    seg["split"] = np.where(seg["train"] == 1, "train", "test")

    test = pd.read_csv(proc / "test_predictions.csv").set_index("segment")
    for m in SCORED:
        thr = metrics["cv"][m]["threshold_policy"]["threshold"]
        oof = pd.read_csv(proc / "oof" / f"{m}.csv").set_index("segment")["score"]
        score = seg["segment"].map(oof).fillna(seg["segment"].map(test[f"{m}_score"]))
        seg[f"{m}_score"] = score
        seg[f"{m}_flag"] = (score >= thr).astype(int)

    # Novelty percentile: share of train-normal (out-of-fold) PCA scores below this segment's.
    ref = np.sort(seg.loc[(seg["train"] == 1) & (seg["anomaly"] == 0), "pca_recon_score"])
    below = np.searchsorted(ref, seg["pca_recon_score"], side="right")
    seg["pca_recon_pct"] = 100 * below / len(ref)
    seg["score_source"] = np.where(seg["train"] == 1, "out-of-fold (grouped CV)",
                                   "final model (frozen test)")
    assert seg[[f"{m}_score" for m in SCORED]].notna().all().all(), "missing scores"
    return seg.drop(columns="train").sort_values("segment").reset_index(drop=True)


def shap_table(cfg: dict) -> pd.DataFrame:
    """Held-out SHAP for train segments + final-model SHAP for test segments (log-odds units)."""
    proc = cfg["paths"]["processed"]
    oof = pd.read_csv(proc / "shap" / "hgb_oof.csv")
    oof.insert(5, "shap_source", "held-out fold model")

    df = load_modeling_table(cfg)
    test = df[df["train"] == 0].reset_index(drop=True)
    X, y, _ = split_xy(test)
    pipe = joblib.load(PROJECT_ROOT / "models" / "hgb.joblib")["model"]
    sv, base, _ = shap_frame(pipe, X)
    logit = pipe.named_steps["clf"].decision_function(pipe.named_steps["pre"].transform(X))
    sv.insert(0, "shap_source", "final model")
    sv.insert(0, "base_value", base)
    sv.insert(0, "logit", logit)
    sv.insert(0, "anomaly", y)
    sv.insert(0, "channel_id", test["channel"])
    sv.insert(0, "segment", test["segment"])
    sv = sv[oof.columns]
    out = pd.concat([oof, sv.round(5)], ignore_index=True).sort_values("segment")
    return out.reset_index(drop=True)


def feature_table(cfg: dict) -> pd.DataFrame:
    df = load_modeling_table(cfg)
    return df[["segment", *numeric_features(df)]]


def trace_table(cfg: dict) -> pd.DataFrame:
    """Raw samples as (segment, seconds since segment start, value)."""
    s = pd.read_csv(cfg["paths"]["raw"] / "segments.csv", usecols=["segment", "timestamp", "value"],
                    parse_dates=["timestamp"])
    s = s.sort_values(["segment", "timestamp"])
    t0 = s.groupby("segment")["timestamp"].transform("min")
    s["t_s"] = (s["timestamp"] - t0).dt.total_seconds().astype(int)
    return s[["segment", "t_s", "value"]]


def main() -> None:
    cfg = load_config()
    out = cfg["paths"]["processed"] / "dashboard"
    out.mkdir(parents=True, exist_ok=True)
    seg = segment_table(cfg)
    seg.to_csv(out / "segments.csv", index=False, float_format="%.6g")
    sh = shap_table(cfg)
    assert set(sh["segment"]) == set(seg["segment"])
    sh.to_csv(out / "shap.csv", index=False)
    feature_table(cfg).to_csv(out / "features.csv", index=False, float_format="%.6g")
    trace_table(cfg).to_csv(out / "traces.csv.gz", index=False, float_format="%.6g")
    for f in sorted(out.iterdir()):
        print(f"{f.name:<16} {f.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
