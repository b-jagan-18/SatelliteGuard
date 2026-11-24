import json

import pandas as pd
import plotly.express as px
import streamlit as st
from common import ROOT, audit, report, scope_note, segments

st.title("Data health")
scope_note()

A, L, seg = audit(), report("leakage_check"), segments()

# ---------------------------------------------------------------- contract
st.subheader("Data contract")
checks = pd.DataFrame([{"check": k.replace("_", " "), "status": "✅ pass" if v["passed"]
                        else "❌ fail"} for k, v in A["checks"].items()])
c1, c2 = st.columns([1, 2])
c1.metric("Checks passed", f"{int(sum(v['passed'] for v in A['checks'].values()))} / "
          f"{len(A['checks'])}")
c1.caption("Run `python -m satelliteguard.data.validate` to regenerate. Includes checksum-"
           "verified download, publisher split counts, and that the raw `label` column is a "
           "constant (the numeric `anomaly` column is the target).")
c2.dataframe(checks, hide_index=True, width="stretch", height=300)

# ---------------------------------------------------------------- balance
st.subheader("Class balance")
counts = seg.groupby(["split", "benchmark_label"]).size().unstack(fill_value=0)
counts["anomaly rate"] = counts["anomalous"] / counts.sum(axis=1)
st.dataframe(counts, width="content",
             column_config={"anomaly rate": st.column_config.NumberColumn(format="%.3f")})
by_ch = seg.groupby("channel").agg(segments=("segment", "size"), rate=("anomaly", "mean"))
fig = px.bar(by_ch.reset_index(), x="channel", y="rate", text="segments",
             labels={"rate": "anomaly rate", "segments": "segments"})
fig.update_traces(marker_color="#DD5555", texttemplate="n=%{text}", textposition="outside")
fig.update_layout(height=330, yaxis_range=[0, 1], margin={"l": 10, "r": 10, "t": 10, "b": 10})
st.plotly_chart(fig, width="stretch")
st.caption("Anomaly rate differs strongly by channel. A channel-only baseline (PR-AUC ≈ 0.25 in "
           "CV) is reported so that feature models must beat 'which channel is it?'.")

# ---------------------------------------------------------------- leakage
st.subheader("Neighbour leakage")
test = seg[seg["split"] == "test"]
c = st.columns(4)
c[0].metric("Contiguous episodes", seg["episode"].nunique())
c[1].metric("Test segments touching train", f"{test['touches_other_split'].mean():.0%}")
c[2].metric("Random-CV PR-AUC", f"{L['random']['model_pr_auc']['mean']:.3f}")
c[3].metric("Episode-grouped CV PR-AUC", f"{L['grouped']['model_pr_auc']['mean']:.3f}",
            f"{-L['optimism_pr_auc']:+.3f} vs random", delta_color="off")
st.markdown("Segments are back-to-back slices of continuous telemetry. Shuffling them into "
            "random folds puts near-identical neighbours on both sides of the split and inflates "
            "scores. SatelliteGuard groups segments into **episodes** (gap ≤ 10 s on the same "
            "channel) and never splits an episode across CV folds.")
st.image(str(ROOT / "reports" / "figures" / "02_segment_timeline.png"), width="stretch")

# ---------------------------------------------------------------- provenance
st.subheader("Provenance")
manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
st.markdown("Ruszczak, B. et al. **OPSSAT-AD — anomaly detection dataset for satellite "
            "telemetry**. Zenodo record [15108715](https://zenodo.org/records/15108715), "
            "CC-BY-4.0. Raw files are re-downloaded and checksum-verified; hashes below.")
with st.expander("data/manifest.json"):
    st.json(manifest)
