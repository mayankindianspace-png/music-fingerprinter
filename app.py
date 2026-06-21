"""
app.py  —  Q3B: Zapp tain America
Shazam-style music identifier — Streamlit UI
Compatible with Streamlit 1.58+ / Python 3.14
"""

import streamlit as st
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, os, csv, tempfile
from collections import defaultdict
from pathlib import Path

# ── resolve paths relative to this file, works on Streamlit Cloud too ──
BASE_DIR  = Path(__file__).parent
SONGS_DIR = BASE_DIR / "songs"
DB_PATH   = BASE_DIR / "fingerprint_db.pkl"

import sys
sys.path.insert(0, str(BASE_DIR))

from fingerprint import (
    SR, load_audio, compute_spectrogram,
    get_constellation, generate_hashes,
    FingerprintDB,
    plot_spectrogram, plot_constellation, plot_offset_histogram,
    fig_to_bytes,
)

# ─────────────────────────────────────────────────────────────
#  Page config & CSS
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zapp tain America • Music Fingerprinter",
    page_icon="🎵",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0a0a14; color: #e0e0f0; }
[data-testid="stSidebar"]          { background: #10101e; border-right: 1px solid #2a2a4a; }
h1 { color: #00c8ff; letter-spacing: 1px; }
h2, h3 { color: #a0c4ff; }
.stButton > button {
    background: linear-gradient(135deg, #00c8ff 0%, #5b5bff 100%);
    color: #fff; border: none; border-radius: 8px;
    font-weight: 600; padding: 0.5rem 1.5rem;
}
.match-box {
    background: #0d2a0d; border: 2px solid #00ff88;
    border-radius: 10px; padding: 1.2rem 1.6rem; margin-top: 0.8rem;
}
.no-match-box {
    background: #2a0d0d; border: 2px solid #ff4444;
    border-radius: 10px; padding: 1.2rem 1.6rem; margin-top: 0.8rem;
}
.step-label {
    font-size: 0.78rem; color: #888;
    text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  Database — build once, cache for the session
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Indexing song database…")
def get_database():
    db = FingerprintDB()
    if DB_PATH.exists():
        try:
            db.load(str(DB_PATH))
            if db.song_lengths:
                return db
        except Exception:
            pass   # corrupt pickle → rebuild
    if SONGS_DIR.exists():
        song_files = sorted(SONGS_DIR.glob("*.mp3")) + \
                     sorted(SONGS_DIR.glob("*.wav"))
        for p in song_files:
            try:
                db.index_from_file(str(p), sr=SR)
            except Exception as e:
                st.warning(f"Could not index {p.name}: {e}")
        try:
            db.save(str(DB_PATH))
        except Exception:
            pass   # read-only filesystem on some hosts — just continue
    return db

db = get_database()

# ─────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎵 Zapp tain America")
    st.caption("EE200 Course Project — Q3B")
    st.markdown("---")
    mode = st.radio("**Mode**", ["Single Clip", "Batch"])
    st.markdown("---")
    st.markdown("**Indexed songs**")
    if db.song_lengths:
        for name in sorted(db.song_lengths):
            st.markdown(f"• `{name}`")
    else:
        st.warning("No songs found in `songs/` folder.")
    st.markdown("---")
    st.caption(f"DB: {len(db.db):,} hashes · {len(db.song_lengths)} songs")

# ─────────────────────────────────────────────────────────────
#  Helper — identify one clip
# ─────────────────────────────────────────────────────────────
def run_identification(audio_bytes: bytes, filename: str):
    suffix = Path(filename).suffix or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        audio = load_audio(tmp_path, sr=SR)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    S_db, freqs, times = compute_spectrogram(audio)
    constellation      = get_constellation(S_db)
    best, offset_hist, scores = db.match(audio)

    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#0d0d0d', 'axes.facecolor': '#12122a',
        'axes.edgecolor': '#333',      'axes.labelcolor': '#ccc',
        'xtick.color': '#999',         'ytick.color': '#999',
        'text.color':  '#ddd',
    })

    fig_spec, ax1 = plt.subplots(figsize=(9, 3.5))
    plot_spectrogram(S_db, freqs, times, title="Spectrogram of Query Clip", ax=ax1)
    plt.tight_layout()

    fig_const, ax2 = plt.subplots(figsize=(9, 3.5))
    plot_constellation(S_db, freqs, times, constellation,
                       title=f"Constellation Map ({len(constellation)} peaks)", ax=ax2)
    plt.tight_layout()

    fig_hist, ax3 = plt.subplots(figsize=(9, 3))
    if best:
        plot_offset_histogram(offset_hist, best,
                              title=f"Offset Histogram — Best: '{best}'", ax=ax3)
    else:
        ax3.text(0.5, 0.5, "No match found", ha='center', va='center',
                 transform=ax3.transAxes, color='#ff4444', fontsize=13)
        ax3.set_title("Offset Histogram")
    plt.tight_layout()

    return best, scores, fig_spec, fig_const, fig_hist

# ─────────────────────────────────────────────────────────────
#  MODE 1 — Single Clip
# ─────────────────────────────────────────────────────────────
if mode == "Single Clip":
    st.title("🎵 Zapp tain America — Single Clip Identifier")
    st.markdown(
        "Upload a short audio clip (MP3 or WAV). The system will fingerprint it "
        "and identify which song it belongs to, showing every intermediate step."
    )

    uploaded = st.file_uploader("Upload query clip", type=["mp3", "wav"])

    if uploaded:
        st.audio(uploaded)
        if st.button("🔍 Identify Song"):
            if not db.song_lengths:
                st.error("Database is empty. No songs found in the `songs/` folder.")
            else:
                with st.spinner("Fingerprinting and matching…"):
                    best, scores, fig_spec, fig_const, fig_hist = \
                        run_identification(uploaded.read(), uploaded.name)

                if best:
                    st.markdown(
                        f'<div class="match-box">'
                        f'<span style="font-size:1.6rem">🎵</span> &nbsp;'
                        f'<strong style="font-size:1.3rem; color:#00ff88">'
                        f'Matched: {best}</strong>'
                        f'<br><span style="color:#88cc88; font-size:0.85rem">'
                        f'Score: {scores.get(best, 0)} aligned hashes</span>'
                        f'</div>', unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div class="no-match-box">'
                        '⚠️ <strong>No match found</strong> in the database.'
                        '</div>', unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("### Intermediate Steps")

                st.markdown('<p class="step-label">Step 1 — Spectrogram</p>',
                            unsafe_allow_html=True)
                st.image(fig_to_bytes(fig_spec), use_container_width=True)

                st.markdown('<p class="step-label">Step 2 — Constellation Map</p>',
                            unsafe_allow_html=True)
                st.image(fig_to_bytes(fig_const), use_container_width=True)

                st.markdown('<p class="step-label">Step 3 — Offset Histogram</p>',
                            unsafe_allow_html=True)
                st.image(fig_to_bytes(fig_hist), use_container_width=True)

                if scores:
                    st.markdown("### All Song Scores")
                    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
                    cols = st.columns(min(len(sorted_scores), 3))
                    for i, (sname, sc) in enumerate(sorted_scores):
                        with cols[i % len(cols)]:
                            st.metric(label=sname, value=sc,
                                      delta="✓ Best match" if sname == best else "")

# ─────────────────────────────────────────────────────────────
#  MODE 2 — Batch
# ─────────────────────────────────────────────────────────────
else:
    st.title("📂 Zapp tain America — Batch Identification")
    st.markdown(
        "Upload multiple query clips. Results are exported as "
        "`results.csv` with columns `filename, prediction`."
    )

    uploaded_files = st.file_uploader(
        "Upload query clips", type=["mp3", "wav"], accept_multiple_files=True
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) queued.**")
        if st.button("▶ Run Batch Identification"):
            if not db.song_lengths:
                st.error("Database is empty.")
            else:
                rows = []
                progress = st.progress(0, text="Processing…")
                for i, uf in enumerate(uploaded_files):
                    try:
                        best, *_ = run_identification(uf.read(), uf.name)
                        prediction = best if best else "NO_MATCH"
                    except Exception as e:
                        prediction = "ERROR"
                        st.warning(f"{uf.name}: {e}")
                    rows.append({"filename": uf.name, "prediction": prediction})
                    progress.progress((i + 1) / len(uploaded_files),
                                      text=f"{i+1}/{len(uploaded_files)} done")

                import pandas as pd
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

                csv_buf = io.StringIO()
                writer  = csv.DictWriter(csv_buf, fieldnames=["filename", "prediction"])
                writer.writeheader()
                writer.writerows(rows)
                st.download_button(
                    label="⬇ Download results.csv",
                    data=csv_buf.getvalue().encode(),
                    file_name="results.csv",
                    mime="text/csv",
                )
                progress.empty()
