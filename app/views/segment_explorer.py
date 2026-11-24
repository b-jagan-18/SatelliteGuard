import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import LABEL_COLORS, features, scope_note, segment_label, segments, shap_values, traces
from common import threshold as policy_threshold

from satelliteguard.dashboard.queries import episode_context, top_contributions

st.title("Segment explorer")
scope_note()

seg, shap_df, feats, tr = segments(), shap_values(), features(), traces()
labels = {int(r.segment): segment_label(r) for r in seg.itertuples()}
ids = list(labels)
default = st.session_state.get("segment", int(seg.loc[seg["split"] == "test"].sort_values(
    "hgb_score", ascending=False)["segment"].iloc[0]))

c1, c2 = st.columns([2, 1])
sid = c1.selectbox("Segment", ids, index=ids.index(default), format_func=labels.get,
                   help="Type a segment id to search.")
show_labels = c2.toggle("Show benchmark labels", value=True)
st.session_state["segment"] = sid
row = seg.loc[seg["segment"] == sid].iloc[0]
thr_h, thr_p = policy_threshold("hgb"), policy_threshold("pca_recon")

# ---------------------------------------------------------------- scores
c = st.columns(5)
c[0].metric("Boosting probability", f"{row.hgb_score:.3f}",
            "flagged" if row.hgb_flag else "not flagged",
            delta_color="inverse" if row.hgb_flag else "off",
            help=f"Primary model. Flag threshold {thr_h:.3f}.")
c[1].metric("Novelty percentile", f"{row.pca_recon_pct:.1f}",
            "flagged" if row.pca_recon_flag else "not flagged",
            delta_color="inverse" if row.pca_recon_flag else "off",
            help="PCA reconstruction (no labels used in fitting): share of normal training "
            "segments that reconstruct better than this one.")
c[2].metric("Logistic probability", f"{row.logreg_score:.3f}", help="Interpretable baseline.")
c[3].metric("Channel", row.channel, row.split, delta_color="off", delta_arrow="off")
if show_labels:
    c[4].metric("Benchmark label", row.benchmark_label)
st.caption(f"Scores: {row.score_source}. {int(row.n_samples)} samples at {int(row.sampling)} s "
           f"sampling, {row.start:%Y-%m-%d %H:%M:%S} → {row.end:%H:%M:%S} UTC. "
           f"Episode {int(row.episode)}, position {int(row.pos_in_episode)}.")

# ---------------------------------------------------------------- trace
st.subheader("Telemetry trace")
width = st.slider("Neighbouring segments to show on each side (same contiguous episode)", 0, 5, 2)
ctx = episode_context(seg, sid, width)
fig = go.Figure()
for r in ctx.itertuples():
    t = tr[tr["segment"] == r.segment]
    x = r.start + pd.to_timedelta(t["t_s"], unit="s")
    current = r.segment == sid
    color = LABEL_COLORS[r.benchmark_label] if show_labels else "#555555"
    name = f"{int(r.segment)}" + (f" ({r.benchmark_label})" if show_labels else "")
    fig.add_trace(go.Scatter(
        x=x, y=t["value"], mode="lines", name=name + (" ◀ selected" if current else ""),
        line={"color": color, "width": 2.5 if current else 1.2},
        opacity=1.0 if current else 0.45,
        hovertemplate=f"segment {int(r.segment)}<br>%{{x}}<br>value %{{y:.4g}}<extra></extra>"))
fig.add_vrect(x0=row.start, x1=row.end, fillcolor="#FFD54F", opacity=0.15, line_width=0)
fig.update_layout(height=340, margin={"l": 10, "r": 10, "t": 10, "b": 10},
                  xaxis_title="time (UTC)", yaxis_title="value (raw units)",
                  legend={"orientation": "h", "y": -0.25})
st.plotly_chart(fig, width="stretch")
st.caption("Neighbours are shown for context only; each segment is scored from its own features.")

# ---------------------------------------------------------------- why flagged
st.subheader("Why did the model score it this way?")
srow = shap_df.loc[shap_df["segment"] == sid].iloc[0]
contrib = top_contributions(srow, k=8)
sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
p_base, p_model = sig(srow["base_value"]), sig(srow["logit"])

left, right = st.columns([3, 2])
with left:
    d = contrib.iloc[::-1]
    bar = go.Figure(go.Bar(
        x=d["contribution"], y=d["feature"], orientation="h",
        marker_color=np.where(d["contribution"] >= 0, "#DD5555", "#4C72B0"),
        hovertemplate="%{y}: %{x:+.2f} log-odds<extra></extra>"))
    bar.update_layout(height=360, margin={"l": 10, "r": 10, "t": 30, "b": 10},
                      xaxis_title="contribution (log-odds)  ← normal | anomalous →",
                      title=f"start {p_base:.2f} (average segment) → {p_model:.2f}")
    st.plotly_chart(bar, width="stretch")
with right:
    up = contrib[(contrib["contribution"] > 0) & (contrib["feature"] != "other features")]
    down = contrib[(contrib["contribution"] < 0) & (contrib["feature"] != "other features")]
    st.markdown("**Pushes towards anomalous:** " + (", ".join(f"`{f}`" for f in up["feature"][:3])
                                                    or "none"))
    st.markdown("**Pushes towards normal:** " + (", ".join(f"`{f}`" for f in down["feature"][:3])
                                                 or "none"))
    src = srow["shap_source"]
    st.caption(
        f"SHAP values from the {src} "
        + ("(the fold model that did not see this segment's episode, first CV repeat; its "
           "probability can differ slightly from the 3-repeat average shown above). "
           if src != "final model" else "(fit on the full train split). ")
        + "SHAP explains how the **model** uses features, an association, not a physical cause. "
        "`channel` is the summed effect of the channel indicator.")

# ---------------------------------------------------------------- feature values
st.markdown("**Feature values compared with normal training segments on the same channel**")
ref_ids = seg.loc[(seg["split"] == "train") & (seg["anomaly"] == 0)
                  & (seg["channel"] == row.channel), "segment"]
ref = feats[feats["segment"].isin(ref_ids)]
fv = feats.loc[feats["segment"] == sid].iloc[0]
shown = [f for f in contrib["feature"] if f in feats.columns]
table = pd.DataFrame({
    "feature": shown,
    "value": [fv[f] for f in shown],
    "normal median": [ref[f].median() for f in shown],
    "percentile among normal": [100 * (ref[f] <= fv[f]).mean() if len(ref) else np.nan
                                for f in shown],
})
st.dataframe(table, hide_index=True, width="stretch", column_config={
    "value": st.column_config.NumberColumn(format="%.4g"),
    "normal median": st.column_config.NumberColumn(format="%.4g"),
    "percentile among normal": st.column_config.ProgressColumn(min_value=0, max_value=100,
                                                               format="%.0f")})
st.caption(f"Reference: {len(ref)} normal training segments on {row.channel}.")
