"""Novelty model behaviour, frozen-test lock and test-prediction contract."""

import sys

import numpy as np
import pandas as pd
import pytest

from satelliteguard import load_config
from satelliteguard.models.registry import MODELS, PCAReconstruction


def test_pca_reconstruction_scores_off_subspace_points_as_anomalous():
    rng = np.random.default_rng(0)
    normal = np.c_[rng.normal(size=(200, 2)), np.zeros((200, 2))]  # lives in a 2-D subspace
    m = PCAReconstruction(n_components=2).fit(normal)
    inside = np.array([[1.0, -1.0, 0.0, 0.0]])
    outside = np.array([[1.0, -1.0, 3.0, 3.0]])
    assert m.reconstruction_error(inside)[0] < 1e-8
    assert m.score_samples(outside)[0] < m.score_samples(inside)[0]  # higher = more normal


def test_novelty_specs_are_marked_normal_only():
    for name in ["pca_recon", "iforest"]:
        assert MODELS[name].fit_on_normal_only and not MODELS[name].is_probability


cfg = load_config()
LOCK = cfg["paths"]["metrics"].parent / "test_lock.json"
PREDS = cfg["paths"]["processed"] / "test_predictions.csv"


@pytest.mark.skipif(not LOCK.exists(), reason="frozen test not evaluated yet")
def test_frozen_test_refuses_second_run(monkeypatch):
    from satelliteguard.evaluation import evaluate_test
    monkeypatch.setattr(sys, "argv", ["evaluate_test"])
    with pytest.raises(SystemExit, match="Refusing"):
        evaluate_test.main()


@pytest.mark.skipif(not PREDS.exists(), reason="frozen test not evaluated yet")
def test_test_predictions_cover_test_split_exactly_once():
    preds = pd.read_csv(PREDS)
    dataset = pd.read_csv(cfg["paths"]["raw"] / "dataset.csv")
    assert preds["segment"].is_unique
    assert set(preds["segment"]) == set(dataset.loc[dataset["train"] == 0, "segment"])
