"""Pure functions behind the dashboard views (kept out of Streamlit scripts so they are tested)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

SHAP_META = ["segment", "channel_id", "anomaly", "logit", "base_value", "shap_source"]

AGREEMENT = {
    (1, 1): "both scores flag",
    (1, 0): "boosting only",
    (0, 1): "novelty only",
    (0, 0): "neither",
}

QUEUE_COLUMNS = ["rank", "segment", "split", "channel", "start", "n_samples", "hgb_score",
                 "pca_recon_pct", "agreement", "benchmark_label"]


def agreement(hgb_flag, pca_flag) -> pd.Series:
    return pd.Series([AGREEMENT[(int(a), int(b))] for a, b in zip(hgb_flag, pca_flag, strict=True)],
                     index=hgb_flag.index if isinstance(hgb_flag, pd.Series) else None)


def review_queue(seg: pd.DataFrame, threshold: float, splits=("test",), channels=None,
                 include_novelty_only: bool = False) -> pd.DataFrame:
    """Segments ranked by gradient-boosting probability for operator review.

    A segment enters the queue when the boosting score reaches `threshold`; optionally also when
    only the novelty (PCA) score flags it. Ties are broken by segment id for a stable order.
    """
    d = seg[seg["split"].isin(splits)]
    if channels:
        d = d[d["channel"].isin(channels)]
    flag = d["hgb_score"] >= threshold
    if include_novelty_only:
        flag |= d["pca_recon_flag"] == 1
    q = d[flag].copy()
    q["agreement"] = agreement((q["hgb_score"] >= threshold).astype(int), q["pca_recon_flag"])
    q["benchmark_label"] = np.where(q["anomaly"] == 1, "anomalous", "normal")
    q = q.sort_values(["hgb_score", "segment"], ascending=[False, True])
    q.insert(0, "rank", np.arange(1, len(q) + 1))
    return q[QUEUE_COLUMNS].reset_index(drop=True)


def top_contributions(shap_row: pd.Series, k: int = 8) -> pd.DataFrame:
    """Largest |SHAP| features for one segment; the rest are summed into 'other features'.

    Values are log-odds contributions relative to the model's base value, so
    base_value + contributions.sum() == logit.
    """
    vals = shap_row.drop(labels=[c for c in SHAP_META if c in shap_row.index]).astype(float)
    order = vals.abs().sort_values(ascending=False).index
    top = vals[order[:k]]
    rest = vals[order[k:]].sum()
    out = pd.DataFrame({"feature": top.index, "contribution": top.to_numpy()})
    if len(order) > k:
        out.loc[len(out)] = ["other features", rest]
    out["direction"] = np.where(out["contribution"] >= 0, "towards anomalous", "towards normal")
    return out


def global_importance(shap_df: pd.DataFrame, k: int = 12) -> pd.Series:
    vals = shap_df.drop(columns=[c for c in SHAP_META if c in shap_df.columns])
    return vals.abs().mean().sort_values(ascending=False).head(k)


def pr_curve(y, score) -> tuple[pd.DataFrame, float]:
    p, r, _ = precision_recall_curve(y, score)
    return pd.DataFrame({"recall": r, "precision": p}), float(average_precision_score(y, score))


def confusion_at(y, score, threshold: float) -> dict:
    y, pred = np.asarray(y), (np.asarray(score) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "alerts": tp + fp}


def episode_context(seg: pd.DataFrame, segment: int, width: int = 3) -> pd.DataFrame:
    """Neighbouring segments in the same contiguous episode (for context, not for scoring)."""
    row = seg.loc[seg["segment"] == segment].iloc[0]
    ep = seg[seg["episode"] == row["episode"]]
    near = ep[(ep["pos_in_episode"] - row["pos_in_episode"]).abs() <= width]
    return near.sort_values("pos_in_episode")
