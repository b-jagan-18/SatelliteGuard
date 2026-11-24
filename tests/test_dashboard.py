"""Dashboard query helpers, committed extract contract, and a headless smoke run of every page."""

import sys

import numpy as np
import pandas as pd
import pytest

from satelliteguard import PROJECT_ROOT, load_config
from satelliteguard.dashboard.queries import (
    agreement,
    confusion_at,
    review_queue,
    top_contributions,
)

DASH = PROJECT_ROOT / "data" / "processed" / "dashboard"
APP = PROJECT_ROOT / "app"


def _toy_segments():
    return pd.DataFrame({
        "segment": [1, 2, 3, 4, 5],
        "split": ["test", "test", "test", "train", "test"],
        "channel": ["A", "A", "B", "A", "B"],
        "start": pd.Timestamp("2022-01-01"),
        "n_samples": 10,
        "hgb_score": [0.9, 0.2, 0.9, 0.99, 0.6],
        "pca_recon_flag": [0, 1, 1, 1, 0],
        "pca_recon_pct": [50, 99, 97, 99, 10],
        "anomaly": [1, 0, 1, 1, 0],
    })


def test_review_queue_ranks_by_score_with_stable_ties_and_respects_filters():
    q = review_queue(_toy_segments(), threshold=0.5)
    assert q["segment"].tolist() == [1, 3, 5]  # tie 0.9 broken by id; train row excluded
    assert q["rank"].tolist() == [1, 2, 3]
    assert q["agreement"].tolist() == ["boosting only", "both scores flag", "boosting only"]
    assert review_queue(_toy_segments(), 0.5, channels=["B"])["segment"].tolist() == [3, 5]
    with_novelty = review_queue(_toy_segments(), 0.5, include_novelty_only=True)
    assert 2 in set(with_novelty["segment"])
    assert with_novelty.set_index("segment").loc[2, "agreement"] == "novelty only"


def test_top_contributions_preserve_shap_additivity():
    row = pd.Series({"segment": 1, "channel_id": "A", "anomaly": 1, "logit": 0.0,
                     "base_value": -1.0, "shap_source": "x",
                     "a": 2.0, "b": -0.5, "c": 0.1, "d": -0.05})
    out = top_contributions(row, k=2)
    assert out["feature"].tolist() == ["a", "b", "other features"]
    assert out["contribution"].sum() == pytest.approx(2.0 - 0.5 + 0.1 - 0.05)
    assert out.loc[0, "direction"] == "towards anomalous"


def test_confusion_and_agreement():
    cm = confusion_at([1, 0, 1, 0], [0.9, 0.8, 0.1, 0.2], 0.5)
    assert (cm["tp"], cm["fp"], cm["fn"], cm["tn"]) == (1, 1, 1, 1)
    assert cm["precision"] == cm["recall"] == 0.5
    assert agreement([1, 0], [0, 0]).tolist() == ["boosting only", "neither"]


needs_extract = pytest.mark.skipif(not (DASH / "segments.csv").exists(),
                                   reason="dashboard extract not built")


@needs_extract
def test_extract_covers_every_segment_once_and_matches_frozen_test_flags():
    cfg = load_config()
    seg = pd.read_csv(DASH / "segments.csv")
    idx = pd.read_csv(cfg["paths"]["processed"] / "segment_index.csv")
    assert seg["segment"].is_unique and set(seg["segment"]) == set(idx["segment"])
    preds = pd.read_csv(cfg["paths"]["processed"] / "test_predictions.csv")
    m = seg[seg["split"] == "test"].merge(preds, on="segment", suffixes=("", "_t"))
    for name in ["hgb", "pca_recon", "logreg"]:
        assert (m[f"{name}_flag"] == m[f"{name}_flag_t"]).all()
    oof = pd.read_csv(cfg["paths"]["processed"] / "oof" / "hgb.csv")
    tr = seg[seg["split"] == "train"].merge(oof, on="segment")
    assert np.allclose(tr["hgb_score"], tr["score"], rtol=1e-5)


@needs_extract
def test_extract_shap_is_additive_and_test_shap_matches_final_model_score():
    sh = pd.read_csv(DASH / "shap.csv")
    meta = ["segment", "channel_id", "anomaly", "logit", "base_value", "shap_source"]
    total = sh["base_value"] + sh.drop(columns=meta).sum(axis=1)
    assert np.allclose(total, sh["logit"], atol=1e-3)
    seg = pd.read_csv(DASH / "segments.csv").set_index("segment")
    test = sh[sh["shap_source"] == "final model"].set_index("segment")
    assert len(test) == (seg["split"] == "test").sum()
    prob = 1 / (1 + np.exp(-test["logit"]))
    assert np.allclose(prob, seg.loc[test.index, "hgb_score"], atol=1e-4)


@needs_extract
@pytest.mark.parametrize("page", ["overview", "review_queue", "segment_explorer",
                                  "model_comparison", "data_health"])
def test_dashboard_page_runs_without_exceptions(page, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.syspath_prepend(str(APP))
    at = AppTest.from_file(str(APP / "views" / f"{page}.py"), default_timeout=60).run()
    assert not at.exception, [e.message for e in at.exception]
    sys.modules.pop("common", None)
