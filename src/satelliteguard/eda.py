"""Reusable EDA figures. Each function returns a matplotlib Figure and optionally saves a PNG.

Used by notebooks/01_eda.ipynb; also runnable directly:
    python -m satelliteguard.eda
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

from satelliteguard import load_config  # noqa: E402

NORMAL, ANOM = "#4C72B0", "#DD5555"
TRAIN_C, TEST_C = "#55A868", "#8172B2"
META_COLS = ["segment", "anomaly", "train", "channel", "sampling"]

sns.set_theme(style="whitegrid", context="notebook")


def feature_columns(dataset: pd.DataFrame) -> list[str]:
    return [c for c in dataset.columns if c not in META_COLS]


def _save(fig, out_dir: Path | None, name: str):
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    return fig


def class_balance(dataset: pd.DataFrame, out_dir: Path | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 2.6]})
    split = dataset.assign(split=dataset["train"].map({1: "train", 0: "test"}))
    ct = pd.crosstab(split["split"], split["anomaly"]).loc[["train", "test"]]
    ct.plot.bar(stacked=True, color=[NORMAL, ANOM], ax=axes[0], rot=0)
    for i, (n, a) in enumerate(ct.values):
        axes[0].text(i, n + a + 15, f"{a / (n + a):.1%} anom.", ha="center", fontsize=9)
    axes[0].set(title="Segments by split", xlabel="", ylabel="segments")
    axes[0].legend(["normal", "anomalous"], frameon=False)

    ch = pd.crosstab(dataset["channel"], dataset["anomaly"])
    ch.plot.bar(stacked=True, color=[NORMAL, ANOM], ax=axes[1], rot=30, legend=False)
    rate = dataset.groupby("channel")["anomaly"].mean()
    for i, c in enumerate(ch.index):
        axes[1].text(i, ch.loc[c].sum() + 10, f"{rate[c]:.0%}", ha="center", fontsize=9)
    axes[1].set(title="Segments by channel (label = anomaly rate)", xlabel="", ylabel="")
    fig.tight_layout()
    return _save(fig, out_dir, "01_class_balance")


def segment_timeline(index: pd.DataFrame, out_dir: Path | None = None):
    """Every segment in time order per channel: shows label clustering and the interleaved split.

    The x-axis is the segment's ordinal position within its channel (time-ordered), because the
    data is concentrated in a few days and a calendar axis would hide the structure. Gaps that
    start a new episode are marked with thin black ticks.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True)
    index = index.sort_values(["channel", "start"]).copy()
    index["pos"] = index.groupby("channel").cumcount()
    channels = sorted(index["channel"].unique())
    ypos = {c: i for i, c in enumerate(channels)}
    breaks = index[index["pos_in_episode"] == 0]
    for ax, (col, colors, title) in zip(
        axes,
        [
            ("anomaly", {0: NORMAL, 1: ANOM}, "Benchmark label"),
            ("train", {1: TRAIN_C, 0: TEST_C}, "Publisher split"),
        ],
        strict=True,
    ):
        for val, color in colors.items():
            sub = index[index[col] == val]
            ax.hlines(sub["channel"].map(ypos), sub["pos"] - 0.5, sub["pos"] + 0.5,
                      color=color, lw=10)
        ax.vlines(breaks["pos"] - 0.5, breaks["channel"].map(ypos) - 0.35,
                  breaks["channel"].map(ypos) + 0.35, color="black", lw=0.6)
        ax.set_yticks(range(len(channels)), channels)
        ax.set_ylim(-0.6, len(channels) - 0.4)
        ax.set_title(title, loc="left")
        handles = [plt.Line2D([], [], color=c, lw=6) for c in colors.values()]
        labels = (["normal", "anomalous"] if col == "anomaly" else ["train", "test"])
        ax.legend(handles, labels, loc="upper right", frameon=True, ncol=2)
    axes[1].set_xlabel("segment position within channel (time-ordered); "
                       "black ticks = start of a new contiguous episode")
    fig.suptitle("Anomalies cluster in runs, and train/test segments are interleaved "
                 "within the same episodes", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "02_segment_timeline")


def length_duration(dataset: pd.DataFrame, out_dir: Path | None = None):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes[:2], ["len", "duration"], strict=True):
        for val, color, lab in [(0, NORMAL, "normal"), (1, ANOM, "anomalous")]:
            sns.histplot(dataset.loc[dataset["anomaly"] == val, col], ax=ax, color=color,
                         label=lab, log_scale=True, stat="density", element="step", alpha=0.35)
        ax.set_title(f"Segment {col} (log scale)")
        ax.legend(frameon=False)
    sns.countplot(data=dataset, x="sampling", hue="anomaly", palette=[NORMAL, ANOM], ax=axes[2])
    axes[2].set_title("Sampling rate")
    axes[2].legend(["normal", "anomalous"], frameon=False)
    fig.tight_layout()
    return _save(fig, out_dir, "03_length_duration_sampling")


def feature_distributions(dataset: pd.DataFrame, out_dir: Path | None = None):
    feats = feature_columns(dataset)
    ncols = 6
    nrows = int(np.ceil(len(feats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 2.8 * nrows))
    for ax, f in zip(axes.flat, feats, strict=False):
        x = dataset[f]
        # symmetric log keeps heavy tails and zeros readable
        lo = np.quantile(np.abs(x[x != 0]), 0.05) if (x != 0).any() else 1.0
        xt = np.sign(x) * np.log10(1 + np.abs(x) / lo)
        for val, color in [(0, NORMAL), (1, ANOM)]:
            sns.kdeplot(xt[dataset["anomaly"] == val], ax=ax, color=color, fill=True,
                        alpha=0.3, warn_singular=False, cut=0)
        ax.set_title(f, fontsize=10)
        ax.set(xlabel="", ylabel="", yticks=[])
    for ax in list(axes.flat)[len(feats):]:
        ax.axis("off")
    fig.suptitle("Feature distributions by class (symlog x-axis; blue = normal, red = anomalous)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "04_feature_distributions")


def correlation_heatmap(dataset: pd.DataFrame, out_dir: Path | None = None):
    feats = feature_columns(dataset)
    corr = dataset[feats].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax,
                annot=True, fmt=".1f", annot_kws={"size": 7}, cbar_kws={"shrink": 0.7})
    ax.set_title("Spearman correlation between features (redundancy check)", loc="left")
    fig.tight_layout()
    return _save(fig, out_dir, "05_feature_correlation")


def univariate_separability(dataset: pd.DataFrame) -> pd.DataFrame:
    """Per-feature ROC-AUC of the raw value vs the label (train split only), plus train/test KS."""
    from sklearn.metrics import roc_auc_score

    tr, te = dataset[dataset["train"] == 1], dataset[dataset["train"] == 0]
    rows = []
    for f in feature_columns(dataset):
        auc = roc_auc_score(tr["anomaly"], tr[f])
        ks = ks_2samp(tr[f], te[f])
        rows.append({"feature": f, "train_auc": auc, "auc_strength": abs(auc - 0.5) * 2,
                     "train_test_ks": ks.statistic, "ks_pvalue": ks.pvalue})
    return pd.DataFrame(rows).sort_values("auc_strength", ascending=False).reset_index(drop=True)


def separability_and_drift(table: pd.DataFrame, out_dir: Path | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    t = table.iloc[::-1]
    axes[0].barh(t["feature"], t["auc_strength"], color=ANOM)
    axes[0].set(title="Univariate separability |2·AUC−1| (train split)", xlim=(0, 1))
    colors = [TEST_C if p < 0.01 else "#BBBBBB" for p in t["ks_pvalue"]]
    axes[1].barh(t["feature"], t["train_test_ks"], color=colors)
    axes[1].set(title="Train vs test drift (KS statistic; purple = p<0.01)")
    fig.tight_layout()
    return _save(fig, out_dir, "06_separability_drift")


def trace_gallery(segments: pd.DataFrame, dataset: pd.DataFrame, n_per_class: int = 4,
                  seed: int = 42, out_dir: Path | None = None):
    """Raw traces for the channels with the most anomalies: n normal + n anomalous each."""
    rng = np.random.default_rng(seed)
    top = dataset.groupby("channel")["anomaly"].sum().sort_values(ascending=False).index[:4]
    fig, axes = plt.subplots(len(top), 2 * n_per_class, figsize=(18, 2.3 * len(top)))
    for r, ch in enumerate(top):
        for k, val in enumerate([0, 1]):
            ids = dataset[(dataset["channel"] == ch) & (dataset["anomaly"] == val)]["segment"]
            pick = rng.choice(ids.values, size=min(n_per_class, len(ids)), replace=False)
            for j, sid in enumerate(pick):
                ax = axes[r, k * n_per_class + j]
                trace = segments[segments["segment"] == sid]
                ax.plot(np.arange(len(trace)), trace["value"].values,
                        color=ANOM if val else NORMAL, lw=1)
                ax.set_xticks([])
                ax.tick_params(labelsize=7)
                ax.set_title(f"{ch} · seg {sid}", fontsize=8)
    fig.suptitle(f"Raw trace gallery — left {n_per_class}: normal, right {n_per_class}: "
                 "anomalous (benchmark labels)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "07_trace_gallery")


def main() -> None:
    cfg = load_config()
    raw, out = cfg["paths"]["raw"], cfg["paths"]["figures"]
    segments = pd.read_csv(raw / "segments.csv")
    dataset = pd.read_csv(raw / "dataset.csv")
    index = pd.read_csv(cfg["paths"]["processed"] / "segment_index.csv",
                        parse_dates=["start", "end"])
    class_balance(dataset, out)
    segment_timeline(index, out)
    length_duration(dataset, out)
    feature_distributions(dataset, out)
    correlation_heatmap(dataset, out)
    table = univariate_separability(dataset)
    separability_and_drift(table, out)
    table.round(4).to_csv(cfg["paths"]["processed"] / "feature_screening.csv", index=False)
    trace_gallery(segments, dataset, out_dir=out)
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
