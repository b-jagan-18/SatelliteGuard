"""Modelling table and leakage-safe preprocessing pipelines.

The modelling table joins the publisher feature table with the episode id. Identifiers, the
target and the split flag never enter X. All transforms are sklearn objects placed inside a
Pipeline, so they are fit on the training fold only.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler

from satelliteguard import load_config

META_COLS = ["segment", "anomaly", "train", "channel", "sampling", "episode"]


def load_modeling_table(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    dataset = pd.read_csv(cfg["paths"]["raw"] / "dataset.csv")
    index = pd.read_csv(cfg["paths"]["processed"] / "segment_index.csv")
    df = dataset.merge(index[["segment", "episode"]], on="segment", validate="one_to_one")
    return df.sort_values("segment").reset_index(drop=True)


def numeric_features(df: pd.DataFrame) -> list[str]:
    """Publisher summary features. `sampling` is kept as a numeric acquisition property."""
    return [c for c in df.columns if c not in META_COLS] + ["sampling"]


def split_xy(df: pd.DataFrame, use_channel: bool = True):
    feats = numeric_features(df)
    cols = (["channel"] if use_channel else []) + feats
    return df[cols], df["anomaly"].to_numpy(), df["episode"].to_numpy()


def make_preprocessor(
    numeric: list[str], scaling: str = "quantile", use_channel: bool = True
) -> ColumnTransformer:
    """Scale numeric features and one-hot the channel.

    scaling="quantile": rank-based map to a normal distribution. Robust to the heavy tails and
        tiny magnitudes (variance ~1e-10) in this table; good for linear/PCA/novelty models.
    scaling="standard": z-score only.
    scaling="none": passthrough (tree models).
    """
    if scaling == "quantile":
        num = QuantileTransformer(output_distribution="normal", n_quantiles=500, random_state=0)
    elif scaling == "standard":
        num = StandardScaler()
    elif scaling == "none":
        num = "passthrough"
    else:
        raise ValueError(scaling)
    transformers = [("num", num, numeric)]
    if use_channel:
        transformers.append(
            ("ch", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["channel"])
        )
    return ColumnTransformer(transformers, verbose_feature_names_out=False)
