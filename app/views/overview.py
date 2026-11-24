import pandas as pd
import streamlit as st
from common import FIGURES, NAMES, SCOPE, report, segments

st.title("SatelliteGuard")
st.markdown("**Leakage-safe anomaly ranking for ESA OPS-SAT telemetry segments (OPSSAT-AD).**")
st.info(SCOPE)

M, lock = report("metrics"), report("test_lock")
T, CV = M["test"]["hgb"], M["cv"]["hgb"]
seg = segments()
test = seg[seg["split"] == "test"]

st.subheader("Primary model on the frozen test split")
c = st.columns(5)
lo, hi = T["bootstrap"]["pr_auc_ci95"]
c[0].metric("Test PR-AUC", f"{T['overall']['pr_auc']:.3f}", help=f"95% episode bootstrap CI "
            f"{lo:.3f}–{hi:.3f}. Chance level = anomaly rate {T['overall']['prevalence']:.2f}.")
c[1].metric("Precision", f"{T['overall']['precision']:.2f}",
            help="Share of flagged test segments that carry the anomalous benchmark label.")
c[2].metric("Recall", f"{T['overall']['recall']:.2f}",
            help="Share of anomalous-labelled test segments that were flagged.")
c[3].metric("Review load", f"{T['overall']['alerts']} / {len(test)}",
            help="Segments flagged at the pre-registered threshold.")
c[4].metric("Grouped-CV PR-AUC", f"{CV['cv_folds']['pr_auc_mean']:.3f}",
            help="Episode-grouped CV on the train split: the honest estimate for unseen episodes.")
st.caption(f"Threshold {CV['threshold_policy']['threshold']:.3f} fixed from train out-of-fold "
           f"predictions ({CV['threshold_policy']['policy']}). Test scored once on "
           f"{lock['evaluated_utc'][:16].replace('T', ' ')} UTC "
           "(lock file `reports/test_lock.json`).")

st.subheader("All models")
rows = []
for m in NAMES:
    cv, te = M["cv"][m], M["test"][m]
    ci = te.get("bootstrap", {}).get("pr_auc_ci95")
    rows.append({
        "model": NAMES[m],
        "track": cv["track"],
        "grouped-CV PR-AUC":
            f"{cv['cv_folds']['pr_auc_mean']:.3f} ± {cv['cv_folds']['pr_auc_std']:.3f}",
        "test PR-AUC": te["overall"]["pr_auc"],
        "test 95% CI": f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "–",
        "test precision": te["overall"]["precision"],
        "test recall": te["overall"]["recall"],
    })
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
             column_config={"test PR-AUC": st.column_config.NumberColumn(format="%.3f"),
                            "test precision": st.column_config.NumberColumn(format="%.2f"),
                            "test recall": st.column_config.NumberColumn(format="%.2f")})

left, right = st.columns([3, 2])
with left:
    st.image(str(FIGURES / "23_cv_vs_test.png"), width="stretch")
with right:
    st.markdown("""
#### Why is test higher than CV?
Segments are back-to-back slices of continuous telemetry, and the publisher's split is
**interleaved**: 89% of test segments touch a training segment. Scores on that split are
optimistic, and almost exactly match a *random* (leaky) CV estimate.

SatelliteGuard selects models with **episode-grouped CV**, which keeps neighbouring segments
together, so **≈0.92 is the honest estimate for new, unseen periods**. The test number is
reported for comparability with the published benchmark.
""")

st.subheader("How to use this dashboard")
st.markdown("""
- **Review queue**: segments ranked by the primary model, with a second-opinion novelty score;
  export as CSV.
- **Segment explorer**: raw trace, neighbouring segments, and *why the model scored it* (SHAP).
- **Model comparison**: precision–recall curves, threshold trade-off, per-channel results,
  significance tests.
- **Data health**: data-contract checks, class balance, leakage analysis, provenance.
""")
st.caption("Data: Ruszczak et al., OPSSAT-AD, Zenodo 15108715, CC-BY-4.0.")
