"""Metrics and the review-queue threshold policy.

Positive class = 1 = anomalous segment. Scores: higher = more anomalous.

Threshold policy (declared before seeing any test data, chosen on out-of-fold CV predictions):
    maximise F1 subject to recall >= MIN_RECALL. Missing an anomaly is assumed costlier than
    reviewing a false alarm, so a recall floor is imposed. This is a *review-queue policy*, not
    an operational safety threshold.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

MIN_RECALL = 0.80
ALERT_BUDGET = 0.25  # reviewers can inspect the top 25% of segments


def select_threshold(y, score, min_recall: float = MIN_RECALL) -> dict:
    """Best-F1 threshold among those with recall >= min_recall."""
    prec, rec, thr = precision_recall_curve(y, score)
    prec, rec = prec[:-1], rec[:-1]  # align with thresholds
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    ok = rec >= min_recall
    if not ok.any():
        ok = np.ones_like(rec, dtype=bool)
    i = np.flatnonzero(ok)[np.argmax(f1[ok])]
    return {"threshold": float(thr[i]), "policy": f"max F1 s.t. recall >= {min_recall}",
            "cv_precision": float(prec[i]), "cv_recall": float(rec[i]), "cv_f1": float(f1[i])}


def recall_at_budget(y, score, budget: float = ALERT_BUDGET) -> tuple[float, float]:
    """Recall and precision if reviewers inspect the top `budget` fraction of segments."""
    y, score = np.asarray(y), np.asarray(score)
    k = max(1, int(round(budget * len(y))))
    top = np.argsort(-score, kind="stable")[:k]
    hits = y[top].sum()
    return float(hits / max(y.sum(), 1)), float(hits / k)


def compute_metrics(y, score, threshold: float | None = None, is_probability: bool = False) -> dict:
    y, score = np.asarray(y), np.asarray(score)
    rb, pb = recall_at_budget(y, score)
    m = {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        f"recall_at_top{int(ALERT_BUDGET * 100)}pct": rb,
        f"precision_at_top{int(ALERT_BUDGET * 100)}pct": pb,
    }
    if is_probability:
        m["brier"] = float(brier_score_loss(y, score))
    if threshold is not None:
        pred = (score >= threshold).astype(int)
        m.update({
            "threshold": float(threshold),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred)),
            "f1": float(f1_score(y, pred)),
            "mcc": float(matthews_corrcoef(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "alerts": int(pred.sum()),
        })
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}
