"""Centralized imports for optional dashboard dependencies.

This module provides a single location for importing optional dashboard
libraries (streamlit, plotly). Other dashboard modules can import from here
to avoid repetitive try/except blocks.
"""

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    st = None
    HAS_STREAMLIT = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    go = None
    HAS_PLOTLY = False

# Dashboard availability requires both streamlit and plotly
HAS_DASHBOARD = HAS_STREAMLIT and HAS_PLOTLY
