"""Frozen-test evaluation. Run ONCE, after model selection is final.

Scores the pre-registered candidates (fit on the full train split, thresholds chosen on train
OOF predictions) on the publisher test split (train=0). Writes:
  reports/metrics.json            -> "test" section
  data/processed/test_predictions.csv
  reports/test_lock.json          -> timestamp, git commit, candidate list

A second run is refused unless --force is given, so test results cannot quietly feed back
into model selection.

Uncertainty: 95% intervals from a cluster bootstrap over test *episodes* (segments in the same
contiguous episode are resampled together, since they are not independent).

Usage:
    python -m satelliteguard.evaluation.evaluate_test
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from satelliteguard import PROJECT_ROOT, load_config
from satelliteguard.evaluation.metrics import compute_metrics
from satelliteguard.features import load_modeling_table, split_xy
from satelliteguard.models.registry import MODELS
from satelliteguard.models.train import update_metrics

CANDIDATES = ["prevalence", "channel_only", "logreg", "hgb", "pca_recon", "iforest"]
PRIMARY = "hgb"
N_BOOT = 2000


def cluster_bootstrap(y, score, thr, groups, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    pr, prec, rec = [], [], []
    for _ in range(n_boot):
        pick = np.concatenate([idx_by_group[g] for g in rng.choice(uniq, len(uniq))])
        yb, sb = y[pick], score[pick]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        pr.append(average_precision_score(yb, sb))
        pred = sb >= thr
        rec.append(pred[yb == 1].mean())
        if pred.any():
            prec.append(yb[pred].mean())
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4),  # noqa: E731
                    round(float(np.percentile(v, 97.5)), 4)]
    return {"pr_auc_ci95": ci(pr), "precision_ci95": ci(prec), "recall_ci95": ci(rec),
            "n_boot": len(pr), "unit": "episode"}


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                      text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                                        text=True).strip()
        return sha + ("+uncommitted" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="override the run-once lock")
    args = parser.parse_args()

    cfg = load_config()
    lock = cfg["paths"]["metrics"].parent / "test_lock.json"
    if lock.exists() and not args.force:
        raise SystemExit(f"Frozen test already evaluated ({lock}). Refusing to re-run.")

    df = load_modeling_table(cfg)
    index = pd.read_csv(cfg["paths"]["processed"] / "segment_index.csv")
    test = df[df["train"] == 0].reset_index(drop=True)
    test = test.merge(index[["segment", "touches_other_split"]], on="segment")
    X, y, groups = split_xy(test)

    preds = test[["segment", "channel", "episode", "anomaly", "touches_other_split"]].copy()
    for name in CANDIDATES:
        bundle = joblib.load(PROJECT_ROOT / "models" / f"{name}.joblib")
        spec = MODELS[name]
        score = spec.score(bundle["model"], X[bundle["features"]])
        thr = bundle["threshold"]
        preds[f"{name}_score"] = score
        preds[f"{name}_flag"] = (score >= thr).astype(int)

        isolated = ~test["touches_other_split"].to_numpy()
        payload = {
            "overall": compute_metrics(y, score, thr, spec.is_probability),
            "bootstrap": cluster_bootstrap(y, score, thr, groups),
            "isolated_segments": (
                compute_metrics(y[isolated], score[isolated], thr)
                if 0 < y[isolated].sum() < isolated.sum() else {"n": int(isolated.sum())}),
            "per_channel": {
                ch: compute_metrics(y[m], score[m], thr) if 0 < y[m].sum() < m.sum()
                else {"n": int(m.sum()), "anomalies": int(y[m].sum()),
                      "flagged": int((score[m] >= thr).sum())}
                for ch in sorted(test["channel"].unique())
                for m in [(test["channel"] == ch).to_numpy()]
            },
        }
        update_metrics(cfg["paths"]["metrics"], "test", name, payload)
        o = payload["overall"]
        print(f"{name:13s} PR-AUC {o['pr_auc']:.3f} {payload['bootstrap']['pr_auc_ci95']} | "
              f"P {o['precision']:.2f} R {o['recall']:.2f} F1 {o['f1']:.2f} MCC {o['mcc']:.2f}")

    preds.to_csv(cfg["paths"]["processed"] / "test_predictions.csv", index=False)
    lock.write_text(json.dumps({
        "evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "candidates": CANDIDATES,
        "primary": PRIMARY,
        "n_test_segments": int(len(test)),
        "forced": args.force,
    }, indent=2), encoding="utf-8")
    print(f"Lock written: {lock}")


if __name__ == "__main__":
    main()
