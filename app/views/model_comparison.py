import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import COLORS, NAMES, report, scope_note, segments, shap_values, threshold

from satelliteguard.dashboard.queries import confusion_at, global_importance, pr_curve

st.title("Model comparison")
scope_note()

M, C, seg = report("metrics"), report("comparisons"), segments()
SCORED = ["hgb", "logreg", "pca_recon"]

# ---------------------------------------------------------------- PR curves
st.subheader("Precision–recall curves")
c1, c2 = st.columns([1, 2])
split = c1.radio("Evaluation set", ["test", "train"], horizontal=True,
                 format_func={"test": "Frozen test (final models)",
                              "train": "Train, out-of-fold (grouped CV)"}.get)
models = c2.multiselect("Models", SCORED, default=SCORED, format_func=NAMES.get)
d = seg[seg["split"] == split]
fig = go.Figure()
prev = d["anomaly"].mean()
fig.add_hline(y=prev, line_dash="dot", line_color="#999999",
              annotation_text=f"chance = anomaly rate {prev:.2f}")
for m in models:
    curve, ap = pr_curve(d["anomaly"], d[f"{m}_score"])
    fig.add_trace(go.Scatter(x=curve["recall"], y=curve["precision"], mode="lines",
                             name=f"{NAMES[m]} (AP {ap:.3f})", line={"color": COLORS[m]}))
    op = confusion_at(d["anomaly"], d[f"{m}_score"], threshold(m))
    fig.add_trace(go.Scatter(x=[op["recall"]], y=[op["precision"]], mode="markers",
                             marker={"color": COLORS[m], "size": 11, "symbol": "x"},
                             showlegend=False, hovertemplate="policy threshold<br>"
                             "recall %{x:.2f}, precision %{y:.2f}<extra></extra>"))
fig.update_layout(height=420, xaxis_title="recall", yaxis_title="precision",
                  xaxis_range=[0, 1.01], yaxis_range=[0, 1.02],
                  margin={"l": 10, "r": 10, "t": 10, "b": 10})
st.plotly_chart(fig, width="stretch")
st.caption("✕ = operating point at the pre-registered threshold. Out-of-fold scores are the mean "
           "over 3 CV repeats, so their AP differs slightly from the per-fold mean in the table.")

# ---------------------------------------------------------------- threshold trade-off
st.subheader("Threshold trade-off (exploratory)")
m = st.selectbox("Model", ["hgb", "logreg"], format_func=NAMES.get)
policy = threshold(m)
thr = st.slider("Threshold", 0.02, 0.98, round(policy, 3), 0.005, key="thr_cmp", format="%.3f")
op = confusion_at(d["anomaly"], d[f"{m}_score"], thr)
c = st.columns(5)
c[0].metric("Precision", f"{op['precision']:.2f}")
c[1].metric("Recall", f"{op['recall']:.2f}")
c[2].metric("Alerts", op["alerts"])
c[3].metric("Missed (FN)", op["fn"])
c[4].metric("False alarms (FP)", op["fp"])
cm = pd.DataFrame([[op["tn"], op["fp"]], [op["fn"], op["tp"]]],
                  index=["labelled normal", "labelled anomalous"],
                  columns=["not flagged", "flagged"])
st.dataframe(cm, width="content")
st.caption(f"Pre-registered threshold {policy:.3f}. Reported results use only that value; "
           "moving the slider is for understanding the trade-off, not for re-tuning on test.")

# ---------------------------------------------------------------- per channel
st.subheader("Primary model per channel (frozen test)")
pc = pd.DataFrame(M["test"]["hgb"]["per_channel"]).T
te = seg[seg["split"] == "test"].groupby("channel")
pc = pd.DataFrame({
    "segments": te.size(),
    "labelled anomalous": te["anomaly"].sum(),
    "flagged": te["hgb_flag"].sum(),
}).join(pc[["pr_auc", "precision", "recall"]])
fmt = {"pr_auc": "{:.3f}", "precision": "{:.2f}", "recall": "{:.2f}"}
for col, f in fmt.items():
    pc[col] = [f.format(v) if pd.notna(v) else "–" for v in pc[col].astype(float)]
st.dataframe(pc.rename(columns={"pr_auc": "PR-AUC"}), width="stretch")
st.caption("Channels with one class in the test split have no ranking metric (counts shown). "
           "Channels with < 10 test segments are anecdotal. CADC0894 is the known weak spot in "
           "both CV and test (long multi-peak normal segments get flagged).")

# ---------------------------------------------------------------- significance
st.subheader("Is the difference real? Paired comparison on identical CV folds")
rows = [{"comparison": f"{NAMES[v['a']]} vs {NAMES[v['b']]}",
         "mean PR-AUC difference": v["mean_diff"],
         "folds A better": f"{v['folds_a_better']} / {v['n_folds']}",
         "naive p": v["p_naive"], "corrected p": v["p_corrected"]}
        for k, v in C.items() if k.endswith("pr_auc")]
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
    "mean PR-AUC difference": st.column_config.NumberColumn(format="%+.3f"),
    "naive p": st.column_config.NumberColumn(format="%.3f"),
    "corrected p": st.column_config.NumberColumn(format="%.3f")})
st.caption("Nadeau–Bengio corrected t-test: CV folds share training data, so the naive t-test is "
           "over-confident. Boosting vs logistic is **not** significant after correction; "
           "boosting vs PCA is.")

# ---------------------------------------------------------------- global SHAP
st.subheader("What the primary model relies on (all segments)")
imp = global_importance(shap_values(), k=12).iloc[::-1]
bar = go.Figure(go.Bar(x=imp.to_numpy(), y=imp.index, orientation="h", marker_color=COLORS["hgb"]))
bar.update_layout(height=380, xaxis_title="mean |SHAP| (log-odds)",
                  margin={"l": 10, "r": 10, "t": 10, "b": 10})
st.plotly_chart(bar, width="stretch")
st.caption("Peak counts and second-difference variance (spikes and abrupt changes) dominate; "
           "channel identity contributes little.")
