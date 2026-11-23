"""Paired model comparison on identical CV folds with the Nadeau-Bengio corrected t-test.

Repeated k-fold scores are not independent (training sets overlap heavily), so a naive paired
t-test is far too confident. Nadeau & Bengio (2003) inflate the variance by (1/J + n_test/n_train)
instead of 1/J, where J is the number of folds evaluated.

Usage:
    python -m satelliteguard.evaluation.compare hgb logreg   # -> reports/comparisons.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

from satelliteguard import load_config


def corrected_paired_ttest(a, b, test_train_ratio: float) -> dict:
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    var = d.var(ddof=1)
    t_naive = d.mean() / np.sqrt(var / J)
    t_corr = d.mean() / np.sqrt(var * (1 / J + test_train_ratio))
    p = lambda t: float(2 * stats.t.sf(abs(t), df=J - 1))  # noqa: E731
    return {"mean_diff": round(float(d.mean()), 4), "sd_diff": round(float(d.std(ddof=1)), 4),
            "folds_a_better": int((d > 0).sum()), "n_folds": J,
            "t_naive": round(float(t_naive), 3), "p_naive": round(p(t_naive), 4),
            "t_corrected": round(float(t_corr), 3), "p_corrected": round(p(t_corr), 4)}


def compare(a: str, b: str, cfg: dict, metric: str = "pr_auc") -> dict:
    d = cfg["paths"]["processed"] / "oof"
    fa, fb = pd.read_csv(d / f"{a}_folds.csv"), pd.read_csv(d / f"{b}_folds.csv")
    assert len(fa) == len(fb), "models must share identical folds"
    k = cfg["cv"]["n_splits"]
    res = corrected_paired_ttest(fa[metric], fb[metric], test_train_ratio=1 / (k - 1))
    return {"a": a, "b": b, "metric": metric, **res}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    args = parser.parse_args()
    cfg = load_config()
    out = cfg["paths"]["metrics"].parent / "comparisons.json"
    data = json.loads(out.read_text()) if out.exists() else {}
    for metric in ["pr_auc", "roc_auc"]:
        r = compare(args.a, args.b, cfg, metric)
        data[f"{args.a}_vs_{args.b}_{metric}"] = r
        print(json.dumps(r))
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
