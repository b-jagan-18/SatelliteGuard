"""Shared loaders and wording for the dashboard. Reads committed extracts; never fits a model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # works without `pip install -e .` (e.g. Streamlit Cloud)
    sys.path.insert(0, str(ROOT / "src"))

DASH = ROOT / "data" / "processed" / "dashboard"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

NAMES = {
    "prevalence": "Prevalence (reference)",
    "channel_only": "Channel only (reference)",
    "logreg": "Logistic regression",
    "hgb": "Gradient boosting (primary)",
    "pca_recon": "PCA reconstruction (novelty)",
    "iforest": "Isolation Forest (novelty)",
}
COLORS = {"prevalence": "#999999", "channel_only": "#C9A227", "logreg": "#4C72B0",
          "hgb": "#DD5555", "pca_recon": "#55A868", "iforest": "#64B5CD"}
LABEL_COLORS = {"anomalous": "#DD5555", "normal": "#4C72B0"}

SCOPE = (
    "SatelliteGuard ranks telemetry **segments for operator review**. A flag means the segment "
    "resembles segments annotated as anomalous in the OPSSAT-AD benchmark. It is **not** a "
    "diagnosis or confirmation of a spacecraft fault, and channel IDs carry no subsystem meaning."
)


def scope_note() -> None:
    st.caption("ⓘ " + SCOPE)


@st.cache_data
def segments() -> pd.DataFrame:
    df = pd.read_csv(DASH / "segments.csv", parse_dates=["start", "end"])
    df["benchmark_label"] = df["anomaly"].map({1: "anomalous", 0: "normal"})
    return df


@st.cache_data
def shap_values() -> pd.DataFrame:
    return pd.read_csv(DASH / "shap.csv")


@st.cache_data
def features() -> pd.DataFrame:
    return pd.read_csv(DASH / "features.csv")


@st.cache_data
def traces() -> pd.DataFrame:
    return pd.read_csv(DASH / "traces.csv.gz")


@st.cache_data
def report(name: str) -> dict:
    return json.loads((REPORTS / f"{name}.json").read_text(encoding="utf-8"))


@st.cache_data
def audit() -> dict:
    return json.loads((ROOT / "data" / "processed" / "data_audit.json").read_text(encoding="utf-8"))


def threshold(model: str = "hgb") -> float:
    return float(report("metrics")["cv"][model]["threshold_policy"]["threshold"])


def segment_label(row) -> str:
    return f"{int(row.segment)} · {row.channel} · {row.split}"
