"""Streamlit entry point for the pre-built DhanDrishti dashboard."""
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(
    page_title="DhanDrishti · Payment Health",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

dashboard_path = Path(__file__).parent / "dashboard" / "index.html"
dashboard_html = dashboard_path.read_text(encoding="utf-8")

# The dashboard is generated and committed ahead of deployment. Rendering this
# static artifact avoids loading the data or fitting the ML model on app startup.
components.html(dashboard_html, height=1850, scrolling=True)
