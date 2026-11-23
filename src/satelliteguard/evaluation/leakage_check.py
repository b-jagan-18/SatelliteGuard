"""Neighbour-leakage experiment: how optimistic is random CV versus episode-grouped CV?

Runs the same fixed model under two CV schemes on the train split only (the frozen test is
never touched here):
  * random  - RepeatedStratifiedKFold (segments assigned to folds independently)
  * grouped - StratifiedGroupKFold with contiguous episodes as groups, repeated with new seeds

Also a channel-only baseline (predict the train-fold anomaly rate of the segment's channel),
which any useful model has to beat.

Usage:
    python -m satelliteguard.evaluation.leakage_check   # writes reports/leakage_check.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from satelliteguard import load_config
from satelliteguard.eda import feature_columns


def make_model(features: list[str], seed: int) -> Pipeline:
    pre = ColumnTransformer(
        [("ch", OneHotEncoder(handle_unknown="ignore"), ["channel"])],
        remainder="passthrough",
    )
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=seed
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def fold_iter(scheme: str, X, y, groups, n_splits: int, n_repeats: int, seed: int):
    if scheme == "random":
        yield from RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed
        ).split(X, y)
    else:
        for r in range(n_repeats):
            yield from StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=seed + r
            ).split(X, y, groups)


def run(cfg: dict) -> dict:
    seed, n_splits, n_repeats = cfg["seed"], cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"]
    dataset = pd.read_csv(cfg["paths"]["raw"] / "dataset.csv")
    index = pd.read_csv(cfg["paths"]["processed"] / "segment_index.csv")
    df = dataset.merge(index[["segment", "episode"]], on="segment", validate="one_to_one")
    df = df[df["train"] == 1].reset_index(drop=True)

    feats = feature_columns(dataset)
    X, y, groups = df[["channel", *feats]], df["anomaly"].values, df["episode"].values

    results = {}
    for scheme in ["random", "grouped"]:
        scores = {"model_pr_auc": [], "model_roc_auc": [], "channel_only_pr_auc": []}
        for tr, va in fold_iter(scheme, X, y, groups, n_splits, n_repeats, seed):
            model = make_model(feats, seed).fit(X.iloc[tr], y[tr])
            p = model.predict_proba(X.iloc[va])[:, 1]
            scores["model_pr_auc"].append(average_precision_score(y[va], p))
            scores["model_roc_auc"].append(roc_auc_score(y[va], p))
            rate = pd.Series(y[tr]).groupby(X["channel"].iloc[tr].values).mean()
            p_ch = X["channel"].iloc[va].map(rate).fillna(y[tr].mean()).values
            scores["channel_only_pr_auc"].append(average_precision_score(y[va], p_ch))
        results[scheme] = {
            k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
            for k, v in scores.items()
        }

    results["optimism_pr_auc"] = round(
        results["random"]["model_pr_auc"]["mean"] - results["grouped"]["model_pr_auc"]["mean"], 4
    )
    results["prevalence"] = round(float(y.mean()), 4)
    results["setup"] = {
        "data": "train split only (train=1); frozen test untouched",
        "model": "HistGradientBoosting (fixed hyperparameters) + one-hot channel",
        "n_splits": n_splits, "n_repeats": n_repeats,
        "groups": "contiguous episodes (gap <= 10 s on same channel)",
    }
    return results


def main() -> None:
    cfg = load_config()
    res = run(cfg)
    out = cfg["paths"]["metrics"].parent / "leakage_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
