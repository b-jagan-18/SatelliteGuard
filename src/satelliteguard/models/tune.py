"""Bounded hyperparameter grids (episode-grouped CV, train split only) + one-SE selection.

Usage:
    python -m satelliteguard.models.tune logreg     # -> reports/tuning/logreg_grid.csv
    python -m satelliteguard.models.tune hgb        # -> reports/tuning/hgb_grid.csv

Choosing the best configuration on the same CV folds makes its CV score slightly optimistic;
grids are kept small (<= 10 configs) to limit this, and the frozen test is the unbiased check.
"""

from __future__ import annotations

import argparse
import itertools

import pandas as pd

from satelliteguard import load_config
from satelliteguard.evaluation.cv import run_cv, summarise_folds
from satelliteguard.features import load_modeling_table, numeric_features, split_xy
from satelliteguard.models.registry import ModelSpec, hgb, iforest, logreg, pca_recon

# family -> (factory builder, grid, simplicity sort: columns + ascending flags, simplest first)
FAMILIES = {
    "logreg": (
        logreg,
        {"C": [0.01, 0.1, 1.0, 10.0, 100.0], "scaling": ["quantile", "standard"]},
        (["C"], [True]),  # smaller C = stronger regularisation = simpler
    ),
    "hgb": (
        hgb,
        {"max_leaf_nodes": [4, 8, 16, 31], "learning_rate": [0.03, 0.1]},
        (["max_leaf_nodes", "learning_rate"], [True, True]),  # fewer leaves, slower learning
    ),
    # Novelty models are fit on train-normal segments only; labels are used only to score folds.
    "iforest": (
        iforest,
        {"max_samples": [128, 256, 512], "max_features": [0.5, 1.0]},
        (["max_samples", "max_features"], [True, True]),  # smaller subsamples = simpler
    ),
    "pca_recon": (
        pca_recon,
        {"n_components": [1, 2, 3, 4, 6, 8, 10]},
        (["n_components"], [True]),
    ),
}
NOVELTY = {"iforest", "pca_recon"}


def one_se_choice(grid: pd.DataFrame, simplicity: tuple[list[str], list[bool]]) -> pd.Series:
    """One-standard-error rule: the simplest config whose mean PR-AUC is within one standard
    error of the best. SE = fold std / sqrt(n_folds) (approximate: folds from different repeats
    are correlated, so the true SE is somewhat larger and this rule is conservative)."""
    best = grid.loc[grid["pr_auc_mean"].idxmax()]
    se = best["pr_auc_std"] / best["n_folds"] ** 0.5
    ok = grid[grid["pr_auc_mean"] >= best["pr_auc_mean"] - se]
    cols, asc = simplicity
    return ok.sort_values(cols, ascending=asc).iloc[0]


def tune(family: str, cfg: dict) -> pd.DataFrame:
    builder, grid_spec, simplicity = FAMILIES[family]
    df = load_modeling_table(cfg)
    train = df[df["train"] == 1].reset_index(drop=True)
    X, y, groups = split_xy(train)
    numeric = numeric_features(train)

    rows = []
    keys = list(grid_spec)
    for values in itertools.product(*grid_spec.values()):
        params = dict(zip(keys, values, strict=True))
        spec = ModelSpec(f"{family}_{params}", builder(**params), "", "supervised",
                         fit_on_normal_only=family in NOVELTY)
        res = run_cv(spec, X, y, groups, numeric, cfg)
        rows.append({**params, **summarise_folds(res["folds"])})
        print(rows[-1])

    out_dir = cfg["paths"]["metrics"].parent / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False)
    choice = one_se_choice(grid, simplicity)
    grid["one_se_choice"] = (grid[keys] == choice[keys]).all(axis=1)
    grid.to_csv(out_dir / f"{family}_grid.csv", index=False)
    print("\nTop 3:\n", grid.head(3).to_string(index=False))
    print("\nOne-SE choice:\n", choice.to_string())
    return grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=list(FAMILIES))
    tune(parser.parse_args().family, load_config())


if __name__ == "__main__":
    main()
