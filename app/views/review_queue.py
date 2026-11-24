import streamlit as st
from common import scope_note, segments, threshold

from satelliteguard.dashboard.queries import review_queue

st.title("Review queue")
st.markdown("Segments ranked by the primary model's probability of carrying the anomalous "
            "benchmark label. The **novelty percentile** is a second opinion from PCA "
            "reconstruction, which never saw a label: *more unusual than X% of normal training "
            "segments*.")
scope_note()

seg = segments()
policy = threshold("hgb")

with st.sidebar:
    st.header("Queue settings")
    splits = st.multiselect("Split", ["test", "train"], default=["test"],
                            help="Test: final model, scored once. Train: out-of-fold CV scores.")
    channels = st.multiselect("Channels", sorted(seg["channel"].unique()),
                              placeholder="All channels")
    thr = st.slider("Probability threshold", 0.05, 0.95, round(policy, 3), 0.005, format="%.3f",
                    help=f"Default {policy:.3f} is the pre-registered policy (max F1 with recall "
                    "≥ 0.80 on train out-of-fold). Changing it is exploratory.")
    novelty = st.checkbox("Also queue novelty-only flags", value=False,
                          help="Add segments that only the PCA novelty score flags.")
    show_labels = st.checkbox("Show benchmark labels", value=True,
                              help="Hide to see the queue as an operator would.")

if not splits:
    st.warning("Select at least one split.")
    st.stop()
if abs(thr - policy) > 1e-3:
    st.warning(f"Threshold {thr:.3f} differs from the pre-registered {policy:.3f}: the numbers "
               "below are exploratory, not the reported results.")

q = review_queue(seg, thr, tuple(splits), channels or None, novelty)
pool = seg[seg["split"].isin(splits) & (seg["channel"].isin(channels) if channels else True)]

c = st.columns(4)
c[0].metric("Segments in queue", f"{len(q)} / {len(pool)}")
c[1].metric("Both scores agree", int((q["agreement"] == "both scores flag").sum()))
if show_labels:
    hits = int((q["benchmark_label"] == "anomalous").sum())
    total = int(pool["anomaly"].sum())
    share = f" ({hits / len(q):.0%})" if len(q) else ""
    c[2].metric("Labelled anomalous in queue", f"{hits}{share}")
    c[3].metric("Labelled anomalous not queued", total - hits)

view = q if show_labels else q.drop(columns="benchmark_label")
event = st.dataframe(
    view, hide_index=True, width="stretch", height=480,
    on_select="rerun", selection_mode="single-row",
    column_config={
        "hgb_score": st.column_config.ProgressColumn("boosting probability", min_value=0.0,
                                                     max_value=1.0, format="%.3f"),
        "pca_recon_pct": st.column_config.ProgressColumn("novelty percentile", min_value=0,
                                                         max_value=100, format="%.1f"),
        "start": st.column_config.DatetimeColumn("start (UTC)", format="YYYY-MM-DD HH:mm"),
        "n_samples": "samples",
    },
)

rows = event.selection.rows
left, right = st.columns(2)
with left:
    if rows:
        chosen = int(q.iloc[rows[0]]["segment"])
        if st.button(f"Open segment {chosen} in explorer", type="primary"):
            st.session_state["segment"] = chosen
            st.switch_page("views/segment_explorer.py")
    else:
        st.caption("Select a row to open it in the segment explorer.")
with right:
    export = q.assign(threshold=round(thr, 4))
    if not show_labels:
        export = export.drop(columns="benchmark_label")
    st.download_button("Download queue (CSV)", export.to_csv(index=False).encode(),
                       file_name=f"satelliteguard_review_queue_thr{thr:.3f}.csv", mime="text/csv")
