"""Episode-grouped, repeated, stratified cross-validation on the train split.

Returns per-fold threshold-free metrics (the CV estimate) and out-of-fold scores averaged over
repeats (used to choose the review threshold and to draw curves). The frozen test split is never
passed in here.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from satelliteguard.models.registry import ModelSpec


def grouped_folds(y, groups, n_splits: int, n_repeats: int, seed: int):
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for tr, va in cv.split(np.zeros(len(y)), y, groups):
            yield r, tr, va


def run_cv(spec: ModelSpec, X, y, groups, numeric: list[str], cfg: dict) -> dict:
    seed, k, reps = cfg["seed"], cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"]
    oof = np.zeros((reps, len(y)))
    folds = []
    template = spec.factory(numeric, seed)
    for r, tr, va in grouped_folds(y, groups, k, reps, seed):
        fit_idx = tr[y[tr] == 0] if spec.fit_on_normal_only else tr
        est = clone(template).fit(X.iloc[fit_idx], y[fit_idx])
        s = spec.score(est, X.iloc[va])
        oof[r, va] = s
        folds.append({
            "repeat": r,
            "pr_auc": average_precision_score(y[va], s),
            "roc_auc": roc_auc_score(y[va], s),
        })
    # Each segment is scored once per repeat (by a model that never saw it); average the repeats.
    oof_mean = oof.mean(axis=0)
    return {"oof": oof_mean, "folds": folds}


def summarise_folds(folds: list[dict]) -> dict:
    out = {}
    for key in ["pr_auc", "roc_auc"]:
        v = np.array([f[key] for f in folds])
        out[f"{key}_mean"] = round(float(v.mean()), 4)
        out[f"{key}_std"] = round(float(v.std()), 4)
    out["n_folds"] = len(folds)
    return out
