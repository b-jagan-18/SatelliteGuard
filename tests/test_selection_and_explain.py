"""Model selection rules, significance test and SHAP consistency."""

import numpy as np
import pandas as pd
import pytest

from satelliteguard import load_config
from satelliteguard.evaluation.compare import corrected_paired_ttest
from satelliteguard.models.tune import one_se_choice


def test_one_se_picks_simplest_within_one_se():
    grid = pd.DataFrame({
        "C": [0.1, 1.0, 10.0],
        "pr_auc_mean": [0.80, 0.885, 0.89],
        "pr_auc_std": [0.04, 0.04, 0.04],
        "n_folds": [16, 16, 16],
    })  # SE = 0.01 -> configs >= 0.88 qualify -> simplest is C = 1
    assert one_se_choice(grid, (["C"], [True]))["C"] == 1.0


def test_corrected_ttest_is_more_conservative():
    rng = np.random.default_rng(0)
    b = rng.normal(0.85, 0.05, 15)
    a = b + rng.normal(0.03, 0.04, 15)
    r = corrected_paired_ttest(a, b, test_train_ratio=0.25)
    assert abs(r["t_corrected"]) < abs(r["t_naive"])
    assert r["p_corrected"] > r["p_naive"]


cfg = load_config()
SHAP = cfg["paths"]["processed"] / "shap" / "hgb_oof.csv"


@pytest.mark.skipif(not SHAP.exists(), reason="run models.explain first")
def test_shap_is_additive_and_out_of_sample():
    sv = pd.read_csv(SHAP)
    contrib = sv.drop(columns=["segment", "channel_id", "anomaly", "logit", "base_value"])
    recon = sv["base_value"] + contrib.sum(axis=1)
    assert np.allclose(recon, sv["logit"], atol=1e-3)  # values are rounded to 5 dp on save
    dataset = pd.read_csv(cfg["paths"]["raw"] / "dataset.csv")
    assert set(sv["segment"]) == set(dataset.loc[dataset["train"] == 1, "segment"])
