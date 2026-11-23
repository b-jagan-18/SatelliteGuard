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
          "logreg_balanced": "#8172B2"}


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


def main() -> None:
    cfg = load_config()
    out = cfg["paths"]["figures"]
    pr_curves(cfg, ["prevalence", "channel_only", "logreg", "logreg_balanced"], out)
    reliability(cfg, ["logreg", "logreg_balanced"], out)
    threshold_tradeoff(cfg, "logreg", out)
    logreg_coefficients(out_dir=out)
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
