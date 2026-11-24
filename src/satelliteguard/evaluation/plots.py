"""Model evaluation figures, generated from saved OOF predictions and metrics.json only.

Usage:
    python -m satelliteguard.evaluation.plots
"""

from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from satelliteguard import PROJECT_ROOT, load_config  # noqa: E402
from satelliteguard.eda import _save  # noqa: E402

COLORS = {"prevalence": "#999999", "channel_only": "#C9A227", "logreg": "#4C72B0",
          "logreg_balanced": "#8172B2", "hgb": "#DD5555", "pca_recon": "#55A868",
          "iforest": "#64B5CD"}


def _oof(cfg, name):
    return pd.read_csv(cfg["paths"]["processed"] / "oof" / f"{name}.csv")


def pr_curves(cfg, names, out_dir=None, fname="10_cv_pr_curves"):
    metrics = json.loads(cfg["paths"]["metrics"].read_text())["cv"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for n in names:
        d = _oof(cfg, n)
        p, r, _ = precision_recall_curve(d["anomaly"], d["score"])
        f = metrics[n]["cv_folds"]
        ax.plot(r, p, color=COLORS.get(n), lw=2,
                label=f"{n}  (fold PR-AUC {f['pr_auc_mean']:.3f} ± {f['pr_auc_std']:.3f})")
    ax.set(xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1.02),
           title="Precision-recall, out-of-fold (episode-grouped CV, train split)")
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    return _save(fig, out_dir, fname)


def reliability(cfg, names, out_dir=None, fname="11_cv_calibration"):
    metrics = json.loads(cfg["paths"]["metrics"].read_text())["cv"]
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
    for n in names:
        d = _oof(cfg, n)
        frac, mean = calibration_curve(d["anomaly"], d["score"], n_bins=10, strategy="quantile")
        brier = metrics[n]["cv_oof_at_threshold"]["brier"]
        ax.plot(mean, frac, "o-", color=COLORS.get(n), label=f"{n} (Brier {brier:.3f})")
    ax.set(xlabel="mean predicted probability", ylabel="observed anomaly rate",
           title="Calibration (reliability) — out-of-fold")
    ax.legend(loc="upper left", fontsize=9)
    return _save(fig, out_dir, fname)


def threshold_tradeoff(cfg, name, out_dir=None, fname="12_cv_threshold_tradeoff"):
    metrics = json.loads(cfg["paths"]["metrics"].read_text())["cv"][name]
    d = _oof(cfg, name)
    p, r, t = precision_recall_curve(d["anomaly"], d["score"])
    alerts = [(d["score"] >= x).mean() for x in t]
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    thr = metrics["threshold_policy"]["threshold"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, p[:-1], label="precision", lw=2)
    ax.plot(t, r[:-1], label="recall", lw=2)
    ax.plot(t, f1, label="F1", lw=2)
    ax.plot(t, alerts, label="fraction of segments flagged", lw=1.5, ls=":")
    ax.axvline(thr, color="black", lw=1)
    ax.text(thr, 0.03, f"  policy threshold {thr:.2f}\n  ({metrics['threshold_policy']['policy']})",
            fontsize=8)
    ax.set(xlabel="score threshold", ylim=(0, 1.02),
           title=f"Review-queue trade-off — {name} (out-of-fold)")
    ax.legend(loc="center right", fontsize=8)
    return _save(fig, out_dir, fname)


def logreg_coefficients(name="logreg", out_dir=None, fname="13_logreg_coefficients"):
    bundle = joblib.load(PROJECT_ROOT / "models" / f"{name}.joblib")
    pipe = bundle["model"]
    names = pipe.named_steps["pre"].get_feature_names_out()
    coef = pd.Series(pipe.named_steps["clf"].coef_[0], index=names).sort_values()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(coef.index, coef.values,
            color=["#DD5555" if v > 0 else "#4C72B0" for v in coef.values])
    ax.axvline(0, color="black", lw=0.8)
    ax.set(title="Logistic coefficients (quantile-normal scaled features)\n"
                 "red = pushes towards 'anomalous'; correlated features share weight",
           xlabel="coefficient")
    return _save(fig, out_dir, fname)


def fold_scatter(cfg, a, b, out_dir=None, fname="20_cv_paired_folds"):
    """Per-fold PR-AUC of model a vs b on identical folds; points above the diagonal favour b."""
    d = cfg["paths"]["processed"] / "oof"
    fa, fb = pd.read_csv(d / f"{a}_folds.csv"), pd.read_csv(d / f"{b}_folds.csv")
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lo = min(fa["pr_auc"].min(), fb["pr_auc"].min()) - 0.02
    ax.plot([lo, 1], [lo, 1], ls="--", color="#999")
    ax.scatter(fa["pr_auc"], fb["pr_auc"], c=fa["repeat"], cmap="viridis", s=50)
    ax.set(xlim=(lo, 1), ylim=(lo, 1), xlabel=f"{a} PR-AUC", ylabel=f"{b} PR-AUC",
           title=f"Paired folds: {b} above diagonal in {(fb['pr_auc'] > fa['pr_auc']).sum()}"
                 f"/{len(fa)}\n(colour = CV repeat)")
    return _save(fig, out_dir, fname)


def pca_components_curve(cfg, out_dir=None, fname="21_pca_components"):
    g = pd.read_csv(cfg["paths"]["metrics"].parent / "tuning" / "pca_recon_grid.csv")
    g = g.sort_values("n_components")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.errorbar(g["n_components"], g["pr_auc_mean"], yerr=g["pr_auc_std"], fmt="o-",
                color=COLORS["pca_recon"], capsize=3)
    ax.set(xlabel="principal components kept", ylabel="PR-AUC (grouped CV)",
           title="PCA reconstruction: too few components underfit normal behaviour,\n"
                 "too many also reconstruct the anomalies")
    return _save(fig, out_dir, fname)


def cv_channel_recall(cfg, names, out_dir=None, fname="22_cv_channel_recall"):
    metrics = json.loads(cfg["paths"]["metrics"].read_text())["cv"]
    rec, fp = {}, {}
    for n in names:
        d = _oof(cfg, n)
        d["flag"] = d["score"] >= metrics[n]["threshold_policy"]["threshold"]
        rec[n] = d[d.anomaly == 1].groupby("channel")["flag"].mean()
        fp[n] = d[d.anomaly == 0].groupby("channel")["flag"].sum()
    rec, fp = pd.DataFrame(rec), pd.DataFrame(fp)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    import seaborn as sns
    sns.heatmap(rec, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=axes[0],
                cbar=False)
    axes[0].set_title("Recall per channel (CV, at each model's policy threshold)")
    sns.heatmap(fp, annot=True, fmt=".0f", cmap="Reds", ax=axes[1], cbar=False)
    axes[1].set_title("False alarms per channel (CV)")
    for ax in axes:
        ax.set(xlabel="", ylabel="")
    fig.tight_layout()
    return _save(fig, out_dir, fname)


def cv_vs_test(cfg, names, out_dir=None, fname="23_cv_vs_test"):
    """Grouped-CV estimate (mean ± sd) vs frozen-test PR-AUC (95% episode bootstrap CI)."""
    root = cfg["paths"]["metrics"].parent
    M = json.loads(cfg["paths"]["metrics"].read_text())
    leak = json.loads((root / "leakage_check.json").read_text())
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(names))
    cv_m = [M["cv"][n]["cv_folds"]["pr_auc_mean"] for n in names]
    cv_s = [M["cv"][n]["cv_folds"]["pr_auc_std"] for n in names]
    te = [M["test"][n]["overall"]["pr_auc"] for n in names]
    ci = np.array([M["test"][n]["bootstrap"]["pr_auc_ci95"] for n in names])
    ax.errorbar(x - 0.12, cv_m, yerr=cv_s, fmt="o", color="#4C72B0", capsize=4,
                label="episode-grouped CV (mean ± sd)")
    ax.errorbar(x + 0.12, te, yerr=[np.array(te) - ci[:, 0], ci[:, 1] - np.array(te)],
                fmt="s", color="#DD5555", capsize=4, label="frozen test (95% episode bootstrap)")
    if "hgb" in names:
        ax.scatter([names.index("hgb")], [leak["random"]["model_pr_auc"]["mean"]], marker="x",
                   s=80, color="black", zorder=5, label="random CV (Day 1, leaky)")
    ax.set_xticks(x, names)
    ax.set(ylabel="PR-AUC", ylim=(0, 1.02),
           title="CV estimate vs frozen test: test sits above grouped CV, near the leaky\n"
                 "random-CV number, because test segments neighbour training segments")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, out_dir, fname)


def test_pr_and_confusion(cfg, names, primary="hgb", out_dir=None, fname="24_test_pr_confusion"):
    from sklearn.metrics import confusion_matrix
    preds = pd.read_csv(cfg["paths"]["processed"] / "test_predictions.csv")
    M = json.loads(cfg["paths"]["metrics"].read_text())["test"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.4, 1]})
    for n in names:
        p, r, _ = precision_recall_curve(preds["anomaly"], preds[f"{n}_score"])
        axes[0].plot(r, p, color=COLORS.get(n), lw=2,
                     label=f"{n} (PR-AUC {M[n]['overall']['pr_auc']:.3f})")
    axes[0].set(xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1.02),
                title="Frozen test — precision-recall")
    axes[0].legend(loc="lower left", fontsize=8)
    cm = confusion_matrix(preds["anomaly"], preds[f"{primary}_flag"])
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[1],
                xticklabels=["not flagged", "flagged"], yticklabels=["normal", "anomalous"])
    axes[1].set(title=f"{primary} at policy threshold (frozen test)", xlabel="model",
                ylabel="benchmark label")
    fig.tight_layout()
    return _save(fig, out_dir, fname)


def main() -> None:
    cfg = load_config()
    out = cfg["paths"]["figures"]
    pr_curves(cfg, ["prevalence", "channel_only", "logreg", "logreg_balanced"], out)
    reliability(cfg, ["logreg", "logreg_balanced"], out)
    threshold_tradeoff(cfg, "logreg", out)
    logreg_coefficients(out_dir=out)
    pr_curves(cfg, ["prevalence", "channel_only", "logreg", "hgb"], out, "18_cv_pr_logreg_vs_hgb")
    reliability(cfg, ["logreg", "hgb"], out, "19_cv_calibration_logreg_vs_hgb")
    fold_scatter(cfg, "logreg", "hgb", out)
    tracks = ["logreg", "hgb", "pca_recon", "iforest"]
    pr_curves(cfg, ["prevalence", *tracks], out, "25_cv_pr_all_tracks")
    pca_components_curve(cfg, out)
    cv_channel_recall(cfg, tracks, out)
    if "test" in json.loads(cfg["paths"]["metrics"].read_text()):
        cv_vs_test(cfg, ["channel_only", *tracks], out)
        test_pr_and_confusion(cfg, ["channel_only", *tracks], out_dir=out)
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
