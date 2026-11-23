"""Cross-validate registered models, choose review thresholds, fit final models.

For each model:
  1. episode-grouped repeated CV on the train split -> fold metrics + out-of-fold (OOF) scores
  2. threshold chosen on OOF scores with the declared policy (metrics.select_threshold)
  3. final model refit on the full train split -> models/<name>.joblib
  4. results merged into reports/metrics.json under "cv" (single source of truth)

The frozen test split is NOT scored here (see evaluation/evaluate_test.py, run once at the end).

Usage:
    python -m satelliteguard.models.train                 # all registered models
    python -m satelliteguard.models.train logreg channel_only
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.base import clone

from satelliteguard import PROJECT_ROOT, load_config
from satelliteguard.evaluation.cv import run_cv, summarise_folds
from satelliteguard.evaluation.metrics import compute_metrics, select_threshold
from satelliteguard.features import load_modeling_table, numeric_features, split_xy
from satelliteguard.models.registry import MODELS


def update_metrics(path, section: str, name: str, payload: dict) -> None:
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data.setdefault(section, {})[name] = payload
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def train_one(name: str, cfg: dict, train: pd.DataFrame) -> dict:
    spec = MODELS[name]
    X, y, groups = split_xy(train)
    numeric = numeric_features(train)

    res = run_cv(spec, X, y, groups, numeric, cfg)
    oof = res["oof"]
    policy = select_threshold(y, oof)
    pooled = compute_metrics(y, oof, policy["threshold"], spec.is_probability)

    oof_dir = cfg["paths"]["processed"] / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"segment": train["segment"], "episode": groups, "channel": train["channel"],
                  "anomaly": y, "score": oof}).to_csv(oof_dir / f"{name}.csv", index=False)

    fit_mask = (y == 0) if spec.fit_on_normal_only else slice(None)
    final = clone(spec.factory(numeric, cfg["seed"])).fit(X[fit_mask], y[fit_mask])
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump({"model": final, "spec_name": name, "threshold": policy["threshold"],
                 "features": list(X.columns)}, model_dir / f"{name}.joblib")

    payload = {
        "description": spec.description,
        "track": spec.track,
        "params": spec.params,
        "fit_on_normal_only": spec.fit_on_normal_only,
        "cv_folds": summarise_folds(res["folds"]),
        "threshold_policy": policy,
        "cv_oof_at_threshold": pooled,
        "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    update_metrics(cfg["paths"]["metrics"], "cv", name, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=list(MODELS))
    names = parser.parse_args().models
    cfg = load_config()
    df = load_modeling_table(cfg)
    train = df[df["train"] == 1].reset_index(drop=True)
    for name in names:
        p = train_one(name, cfg, train)
        f = p["cv_folds"]
        print(f"{name:16s} PR-AUC {f['pr_auc_mean']:.3f} ± {f['pr_auc_std']:.3f} | "
              f"ROC-AUC {f['roc_auc_mean']:.3f} | thr {p['threshold_policy']['threshold']:.3f} "
              f"-> P {p['cv_oof_at_threshold'].get('precision', float('nan')):.2f} "
              f"R {p['cv_oof_at_threshold'].get('recall', float('nan')):.2f}")


if __name__ == "__main__":
    main()
