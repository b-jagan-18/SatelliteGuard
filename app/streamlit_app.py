"""SatelliteGuard dashboard entry point.

Run:  streamlit run app/streamlit_app.py
"""

import streamlit as st

st.set_page_config(page_title="SatelliteGuard", page_icon="🛰️", layout="wide")

pages = [
    st.Page("views/overview.py", title="Overview", icon="🛰️", default=True),
    st.Page("views/review_queue.py", title="Review queue", icon="📋"),
    st.Page("views/segment_explorer.py", title="Segment explorer", icon="🔎"),
    st.Page("views/model_comparison.py", title="Model comparison", icon="📊"),
    st.Page("views/data_health.py", title="Data health", icon="🩺"),
]
st.navigation(pages).run()
