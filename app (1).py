import streamlit as st

st.set_page_config(page_title="Zapp tain America", page_icon="🎵", layout="wide")

# ── Show any import errors directly on screen ──────────────────────────────
import traceback, sys

try:
    import numpy as np
    st.success("✓ numpy")
except Exception as e:
    st.error(f"numpy failed: {e}"); st.stop()

try:
    import librosa
    st.success("✓ librosa")
except Exception as e:
    st.error(f"librosa failed: {e}"); st.stop()

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st.success("✓ matplotlib")
except Exception as e:
    st.error(f"matplotlib failed: {e}"); st.stop()

try:
    from scipy.ndimage import maximum_filter
    st.success("✓ scipy")
except Exception as e:
    st.error(f"scipy failed: {e}"); st.stop()

try:
    import io, os, csv, tempfile, pickle
    from pathlib import Path
    from collections import defaultdict
    st.success("✓ standard library")
except Exception as e:
    st.error(f"stdlib failed: {e}"); st.stop()

st.success("✅ All imports OK — app is working!")
st.write("Python version:", sys.version)
st.write("librosa version:", librosa.__version__)
st.write("numpy version:", np.__version__)
