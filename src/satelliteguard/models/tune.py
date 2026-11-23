"""Bounded hyperparameter grid for the logistic baseline (episode-grouped CV, train split only).

Usage:
    python -m satelliteguard.models.tune      # writes reports/tuning/logreg_grid.csv

Note: choosing the best configuration on the same CV folds makes its CV score slightly
optimistic; the grid is kept small (8 configs) to limit this, and the frozen test is the
unbiased check.
"""

from __future__ import annotations

import itertools

import pandas as pd

from satelliteguard import load_config
from satelliteguard.evaluation.cv import run_cv, summarise_folds
from satelliteguard.features import load_modeling_table, numeric_features, split_xy
from satelliteguard.models.registry import ModelSpec, logreg

GRID = {"C": [0.01, 0.1, 1.0, 10.0, 100.0], "scaling": ["quantile", "standard"]}


def one_se_choice(grid: pd.DataFrame) -> pd.Series:
    """One-standard-error rule: the most regularised (smallest C) config whose mean PR-AUC is
    within one standard error of the best. SE = fold std / sqrt(n_folds) (approximate: folds
    from different repeats are correlated)."""
    best = grid.loc[grid["pr_auc_mean"].idxmax()]
    se = best["pr_auc_std"] / best["n_folds"] ** 0.5
    ok = grid[grid["pr_auc_mean"] >= best["pr_auc_mean"] - se]
    return ok.sort_values(["C", "pr_auc_mean"], ascending=[True, False]).iloc[0]


def main() -> None:
    cfg = load_config()
    df = load_modeling_table(cfg)
    train = df[df["train"] == 1].reset_index(drop=True)
    X, y, groups = split_xy(train)
    numeric = numeric_features(train)

    rows = []
    for C, scaling in itertools.product(GRID["C"], GRID["scaling"]):
        spec = ModelSpec(f"logreg_C{C}_{scaling}", logreg(C=C, scaling=scaling), "", "supervised")
        res = run_cv(spec, X, y, groups, numeric, cfg)
        rows.append({"C": C, "scaling": scaling, **summarise_folds(res["folds"])})
        print(rows[-1])

    out_dir = cfg["paths"]["metrics"].parent / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False)
    grid.to_csv(out_dir / "logreg_grid.csv", index=False)
    print("\nTop 3:\n", grid.head(3).to_string(index=False))
    print("\nOne-SE choice:\n", one_se_choice(grid).to_string())


if __name__ == "__main__":
    main()
