"""Metric correctness, threshold policy, and leakage guards for the CV runner."""

import numpy as np
import pandas as pd
import pytest

from satelliteguard import load_config
from satelliteguard.evaluation.cv import grouped_folds
from satelliteguard.evaluation.metrics import compute_metrics, recall_at_budget, select_threshold
from satelliteguard.models.registry import MODELS


def test_perfect_and_constant_scores():
    y = np.array([0, 0, 0, 1, 1])
    perfect = compute_metrics(y, np.array([0.1, 0.2, 0.3, 0.8, 0.9]), threshold=0.5)
    assert perfect["pr_auc"] == 1.0 and perfect["recall"] == 1.0 and perfect["mcc"] == 1.0
    const = compute_metrics(y, np.full(5, 0.4))
    assert const["pr_auc"] == pytest.approx(y.mean())  # PR-AUC of a constant = prevalence


def test_recall_at_budget():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    rec, prec = recall_at_budget(y, s, budget=0.25)  # top 2 segments
    assert rec == 0.5 and prec == 0.5


def test_threshold_respects_recall_floor():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = y * 0.3 + rng.random(500)
    pol = select_threshold(y, s, min_recall=0.8)
    assert pol["cv_recall"] >= 0.8
    assert compute_metrics(y, s, pol["threshold"])["recall"] >= 0.8


def test_grouped_folds_never_split_an_episode():
    groups = np.repeat(np.arange(40), 5)
    y = (np.arange(200) % 4 == 0).astype(int)
    for _, tr, va in grouped_folds(y, groups, n_splits=5, n_repeats=2, seed=0):
        assert not set(groups[tr]) & set(groups[va])


cfg = load_config()
OOF = cfg["paths"]["processed"] / "oof"


@pytest.mark.skipif(not (OOF / "logreg.csv").exists(), reason="run models.train first")
@pytest.mark.parametrize("name", list(MODELS))
def test_oof_predictions_use_train_split_only(name):
    path = OOF / f"{name}.csv"
    if not path.exists():
        pytest.skip(f"{name} not trained")
    oof = pd.read_csv(path)
    dataset = pd.read_csv(cfg["paths"]["raw"] / "dataset.csv")
    train_ids = set(dataset.loc[dataset["train"] == 1, "segment"])
    assert set(oof["segment"]) == train_ids  # exactly the train split, frozen test untouched
    assert oof["segment"].is_unique
