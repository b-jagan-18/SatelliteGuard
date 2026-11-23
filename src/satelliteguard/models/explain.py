"""SHAP explanations for the gradient-boosting model, computed on HELD-OUT segments.

Each train segment is explained by the fold model that did not see it (first CV repeat), so
explanations describe out-of-sample behaviour. SHAP values are additive in log-odds:
    base_value + sum(shap) = model logit.
The one-hot channel columns are summed into a single "channel" contribution.

SHAP describes how the *model* uses features (association), not physical causes.

Usage:
    python -m satelliteguard.models.explain      # -> data/processed/shap/hgb_oof.csv + figures
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.base import clone  # noqa: E402

from satelliteguard import load_config  # noqa: E402
from satelliteguard.eda import _save  # noqa: E402
from satelliteguard.evaluation.cv import grouped_folds  # noqa: E402
from satelliteguard.features import load_modeling_table, numeric_features, split_xy  # noqa: E402
from satelliteguard.models.registry import MODELS  # noqa: E402


def shap_frame(pipe, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """SHAP values with channel one-hots merged. Returns (shap_df, base_values, feature_values)."""
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    Xt = pre.transform(X)
    names = pre.get_feature_names_out()
    ex = shap.TreeExplainer(clf)(Xt)
    sv = pd.DataFrame(ex.values, columns=names, index=X.index)
    ch_cols = [c for c in names if c.startswith("channel_")]
    sv["channel"] = sv[ch_cols].sum(axis=1)
    sv = sv.drop(columns=ch_cols)
    feats = X[[c for c in sv.columns if c != "channel"]].copy()
    return sv, np.asarray(ex.base_values).ravel(), feats


def explain_oof(name: str, cfg: dict) -> pd.DataFrame:
    spec = MODELS[name]
    df = load_modeling_table(cfg)
    train = df[df["train"] == 1].reset_index(drop=True)
    X, y, groups = split_xy(train)
    numeric = numeric_features(train)
    template = spec.factory(numeric, cfg["seed"])
    parts = []
    for _, tr, va in grouped_folds(y, groups, cfg["cv"]["n_splits"], 1, cfg["seed"]):
        est = clone(template).fit(X.iloc[tr], y[tr])
        sv, base, _ = shap_frame(est, X.iloc[va])
        sv.insert(0, "base_value", base)
        sv.insert(0, "logit", est.named_steps["clf"].decision_function(
            est.named_steps["pre"].transform(X.iloc[va])))
        parts.append(sv)
    out = pd.concat(parts).sort_index()
    out.insert(0, "anomaly", y)
    out.insert(0, "channel_id", train["channel"])
    out.insert(0, "segment", train["segment"])
    return out


def _explanation(sv: pd.DataFrame, feats: pd.DataFrame) -> shap.Explanation:
    cols = [c for c in sv.columns if c not in
            ("segment", "channel_id", "anomaly", "logit", "base_value")]
    data = feats.reindex(columns=cols)
    data["channel"] = np.nan  # categorical: no single numeric value to colour by
    return shap.Explanation(values=sv[cols].to_numpy(), base_values=sv["base_value"].to_numpy(),
                            data=data.to_numpy(dtype=float), feature_names=cols)


def figures(sv: pd.DataFrame, train: pd.DataFrame, out_dir) -> None:
    feats = train.set_index("segment").loc[sv["segment"]].reset_index()
    ex = _explanation(sv.reset_index(drop=True), feats)

    plt.figure()
    shap.plots.bar(ex, max_display=15, show=False)
    fig = plt.gcf()
    fig.suptitle("Global drivers: mean |SHAP| (log-odds), held-out segments",
                 x=0.01, ha="left", fontweight="bold")
    _save(fig, out_dir, "14_shap_global")
    plt.close(fig)

    plt.figure()
    shap.plots.beeswarm(ex, max_display=15, show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP beeswarm: red = high feature value; right = pushes towards anomalous",
                 x=0.01, ha="left", fontweight="bold")
    _save(fig, out_dir, "15_shap_beeswarm")
    plt.close(fig)

    # One confident true positive and the most confidently missed anomaly.
    prob = 1 / (1 + np.exp(-sv["logit"].to_numpy()))
    y = sv["anomaly"].to_numpy()
    tp = int(np.argmax(np.where(y == 1, prob, -1)))
    fn = int(np.argmin(np.where(y == 1, prob, 2)))
    for i, tag, fname in [(tp, "confident detection", "16_shap_waterfall_tp"),
                          (fn, "missed anomaly", "17_shap_waterfall_fn")]:
        plt.figure()
        shap.plots.waterfall(ex[i], max_display=10, show=False)
        fig = plt.gcf()
        fig.suptitle(f"Segment {int(sv['segment'].iloc[i])} ({sv['channel_id'].iloc[i]}) — {tag}, "
                     f"p = {prob[i]:.2f}", x=0.01, ha="left", fontweight="bold")
        _save(fig, out_dir, fname)
        plt.close(fig)


def main(name: str = "hgb") -> None:
    cfg = load_config()
    sv = explain_oof(name, cfg)
    out = cfg["paths"]["processed"] / "shap"
    out.mkdir(parents=True, exist_ok=True)
    sv.round(5).to_csv(out / f"{name}_oof.csv", index=False)
    df = load_modeling_table(cfg)
    figures(sv, df[df["train"] == 1], cfg["paths"]["figures"])
    imp = sv.drop(columns=["segment", "channel_id", "anomaly", "logit", "base_value"]).abs().mean()
    print(imp.sort_values(ascending=False).round(3).head(10).to_string())


if __name__ == "__main__":
    main()
