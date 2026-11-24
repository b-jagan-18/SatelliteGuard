"""Model registry. Every model is a ModelSpec: a factory plus how it is trained and scored.

Adding a model = adding one entry to MODELS. Day 3 adds tree ensembles; Day 4 adds novelty
models with fit_on_normal_only=True.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, OutlierMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from satelliteguard.features import make_preprocessor


@dataclass
class ModelSpec:
    name: str
    factory: Callable[[list[str], int], BaseEstimator]  # (numeric feature names, seed) -> est
    description: str
    track: str  # reference | supervised | unsupervised
    is_probability: bool = True
    fit_on_normal_only: bool = False
    params: dict = field(default_factory=dict)

    def score(self, est, X) -> np.ndarray:
        if hasattr(est, "predict_proba"):
            return est.predict_proba(X)[:, 1]
        return -est.score_samples(X)  # novelty models: higher = more anomalous


class PrevalenceBaseline(ClassifierMixin, BaseEstimator):
    """Scores every segment with the training anomaly rate (PR-AUC = prevalence)."""

    def fit(self, X, y):
        self.rate_ = float(np.mean(y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.rate_)
        return np.column_stack([1 - p, p])


class ChannelRateBaseline(ClassifierMixin, BaseEstimator):
    """Scores a segment with its channel's training anomaly rate. Features must beat this."""

    def fit(self, X, y):
        self.rates_ = pd.Series(y, index=X["channel"].to_numpy()).groupby(level=0).mean()
        self.default_ = float(np.mean(y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = X["channel"].map(self.rates_).fillna(self.default_).to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


def logreg(C: float = 1.0, scaling: str = "quantile", class_weight=None):
    def factory(numeric, seed):
        return Pipeline([
            ("pre", make_preprocessor(numeric, scaling=scaling)),
            ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=5000)),
        ])
    return factory


def hgb(max_leaf_nodes: int = 15, learning_rate: float = 0.05, max_iter: int = 300,
        class_weight=None):
    """Histogram gradient boosting. Fixed iteration count: sklearn's early stopping uses a random
    internal validation split, which would leak neighbouring segments."""
    def factory(numeric, seed):
        return Pipeline([
            ("pre", make_preprocessor(numeric, scaling="none")),
            ("clf", HistGradientBoostingClassifier(
                max_leaf_nodes=max_leaf_nodes, learning_rate=learning_rate, max_iter=max_iter,
                min_samples_leaf=20, l2_regularization=1.0, early_stopping=False,
                class_weight=class_weight, random_state=seed)),
        ])
    return factory


class PCAReconstruction(OutlierMixin, BaseEstimator):
    """Novelty score = squared reconstruction error after projecting onto the top principal
    components of NORMAL training segments. score_samples follows the sklearn convention
    (higher = more normal), so it returns the negative error."""

    def __init__(self, n_components: int = 4):
        self.n_components = n_components

    def fit(self, X, y=None):
        self.pca_ = PCA(n_components=self.n_components).fit(X)
        return self

    def reconstruction_error(self, X):
        X = np.asarray(X, dtype=float)
        rec = self.pca_.inverse_transform(self.pca_.transform(X))
        return ((X - rec) ** 2).sum(axis=1)

    def score_samples(self, X):
        return -self.reconstruction_error(X)


def iforest(n_estimators: int = 300, max_samples="auto", max_features: float = 1.0):
    """Isolation Forest on quantile-scaled numeric features (no channel). `contamination` only
    sets sklearn's internal predict() cut-off; scores come from score_samples and our own
    threshold policy, so contamination does not affect any reported metric."""
    def factory(numeric, seed):
        return Pipeline([
            ("pre", make_preprocessor(numeric, scaling="quantile", use_channel=False)),
            ("clf", IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                                    max_features=max_features, random_state=seed)),
        ])
    return factory


def pca_recon(n_components: int = 4):
    def factory(numeric, seed):
        return Pipeline([
            ("pre", make_preprocessor(numeric, scaling="quantile", use_channel=False)),
            ("clf", PCAReconstruction(n_components=n_components)),
        ])
    return factory


MODELS: dict[str, ModelSpec] = {
    "prevalence": ModelSpec(
        "prevalence", lambda n, s: PrevalenceBaseline(),
        "Constant score = training anomaly rate", "reference"),
    "channel_only": ModelSpec(
        "channel_only", lambda n, s: ChannelRateBaseline(),
        "Training anomaly rate of the segment's channel", "reference"),
    # Selected by a bounded grid (see models/tune.py, reports/tuning/logreg_grid.csv).
    "logreg": ModelSpec(
        "logreg", logreg(C=1.0, scaling="quantile"),
        "L2 logistic regression, quantile-normal scaling + one-hot channel, unweighted "
        "(log-loss fit keeps probabilities calibrated)", "supervised",
        params={"C": 1.0, "scaling": "quantile", "class_weight": None}),
    # Selected by one-SE rule over reports/tuning/hgb_grid.csv.
    "hgb": ModelSpec(
        "hgb", hgb(max_leaf_nodes=4, learning_rate=0.03),
        "Histogram gradient boosting, 4 leaves/tree, lr 0.03, 300 trees, unweighted, raw "
        "features + one-hot channel", "supervised",
        params={"max_leaf_nodes": 4, "learning_rate": 0.03, "max_iter": 300,
                "min_samples_leaf": 20, "l2_regularization": 1.0}),
    # Novelty track: fit on train-normal segments only. Labels are used only to choose the
    # hyperparameter (one-SE rule over reports/tuning/*_grid.csv) and the review threshold.
    "pca_recon": ModelSpec(
        "pca_recon", pca_recon(n_components=3),
        "PCA reconstruction error (3 components) on quantile-scaled features, fit on "
        "train-normal only", "unsupervised", is_probability=False, fit_on_normal_only=True,
        params={"n_components": 3}),
    "iforest": ModelSpec(
        "iforest", iforest(max_samples=128, max_features=1.0),
        "Isolation Forest (300 trees, 128 samples/tree) on quantile-scaled features, fit on "
        "train-normal only", "unsupervised", is_probability=False, fit_on_normal_only=True,
        params={"n_estimators": 300, "max_samples": 128, "max_features": 1.0}),
    "logreg_balanced": ModelSpec(
        "logreg_balanced", logreg(C=1.0, scaling="quantile", class_weight="balanced"),
        "As logreg but class_weight='balanced' (calibration comparison)", "supervised",
        params={"C": 1.0, "scaling": "quantile", "class_weight": "balanced"}),
}
