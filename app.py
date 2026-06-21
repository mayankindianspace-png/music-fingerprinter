"""
app.py  —  Q3B: Zapp tain America
Shazam-style music identifier — Streamlit UI
Two modes:
  1. Single-clip mode : upload a query → show spectrogram, constellation,
                        offset histogram, and matched song name.
  2. Batch mode       : upload multiple clips → download results.csv
"""

import streamlit as st
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io, os, pickle, tempfile, csv
from collections import defaultdict
from pathlib import Path

from fingerprint import (
    SR, N_FFT, HOP_LENGTH,
    load_audio, compute_spectrogram,
    get_constellation, generate_hashes,
    FingerprintDB,
    plot_spectrogram, plot_constellation, plot_offset_histogram,
    fig_to_bytes,
)

# ─────────────────────────────────────────────────────────────
#  Page config & custom CSS
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zapp tain America • Music Fingerprinter",
    page_icon="🎵",
    layout="wide",
)

st.markdown("""
<style>
/* Dark waveform-inspired theme */
[data-testid="stAppViewContainer"] {
    background: #0a0a14;
    color: #e0e0f0;
}
[data-testid="stSidebar"] {
    background: #10101e;
    border-right: 1px solid #2a2a4a;
}
h1 { color: #00c8ff; letter-spacing: 1px; }
h2, h3 { color: #a0c4ff; }
.stButton > button {
    background: linear-gradient(135deg, #00c8ff 0%, #5b5bff 100%);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
}
.stButton > button:hover { opacity: 0.85; }
.match-box {
    background: #0d2a0d;
    border: 2px solid #00ff88;
    border-radius: 10px;
    padding: 1.2rem 1.6rem;
    margin-top: 0.8rem;
}
.no-match-box {
    background: #2a0d0d;
    border: 2px solid #ff4444;
    border-radius: 10px;
    padding: 1.2rem 1.6rem;
    margin-top: 0.8rem;
}
.step-label {
    font-size: 0.78rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  Database — load or build once per session
# ─────────────────────────────────────────────────────────────
DB_PATH   = Path(__file__).parent / "fingerprint_db.pkl"
SONGS_DIR = Path(__file__).parent / "songs"

@st.cache_resource(show_spinner="Indexing song database…")
def get_database():
    db = FingerprintDB()
    if DB_PATH.exists():
        db.load(str(DB_PATH))
        if db.song_lengths:           # already has songs
            return db
    # build from scratch
    if SONGS_DIR.exists():
        song_files = sorted(SONGS_DIR.glob("*.mp3")) + \
                     sorted(SONGS_DIR.glob("*.wav"))
        for p in song_files:
            db.index_from_file(str(p), sr=SR)
        db.save(str(DB_PATH))
    return db

db = get_database()

# ─────────────────────────────────────────────────────────────
#  Sidebar — navigation & database info
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎵 Zapp tain America")
    st.caption("EE200 Course Project — Q3B")
    st.markdown("---")

    mode = st.radio("**Mode**", ["Single Clip", "Batch"])
    st.markdown("---")

    # Show indexed songs
    st.markdown("**Indexed songs**")
    if db.song_lengths:
        for name in sorted(db.song_lengths):
            st.markdown(f"• `{name}`")
    else:
        st.warning("No songs indexed yet. Place `.mp3` / `.wav` files in `songs/`.")

    st.markdown("---")
    st.caption(f"DB keys: {len(db.db):,} hashes · {len(db.song_lengths)} songs")


# ─────────────────────────────────────────────────────────────
#  Helper — run match and return figures
# ─────────────────────────────────────────────────────────────
def run_identification(audio_bytes: bytes, filename: str):
    """
    Load audio from bytes, fingerprint it, match against DB.
    Returns (best_match, scores, fig_spec, fig_const, fig_hist)
    """
    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        audio = load_audio(tmp_path, sr=SR)
    finally:
        os.unlink(tmp_path)

    # Spectrogram
    S_db, freqs, times = compute_spectrogram(audio)
    # Constellation
    constellation = get_constellation(S_db)
    # Match
    best, offset_hist, scores = db.match(audio)

    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#0d0d0d', 'axes.facecolor': '#12122a',
        'axes.edgecolor': '#333', 'axes.labelcolor': '#ccc',
        'xtick.color': '#999', 'ytick.color': '#999',
        'text.color': '#ddd',
    })

    # -- Figure 1: spectrogram --
    fig_spec, ax1 = plt.subplots(figsize=(9, 3.5))
    plot_spectrogram(S_db, freqs, times, title="Spectrogram of Query Clip", ax=ax1)
    plt.tight_layout()

    # -- Figure 2: constellation --
    fig_const, ax2 = plt.subplots(figsize=(9, 3.5))
    plot_constellation(S_db, freqs, times, constellation,
                       title=f"Constellation Map ({len(constellation)} peaks)", ax=ax2)
    plt.tight_layout()

    # -- Figure 3: offset histogram --
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
        st.audio(uploaded, format=f"audio/{Path(uploaded.name).suffix.lstrip('.')}")
        col_btn, _ = st.columns([1, 4])
        with col_btn:
            identify = st.button("🔍 Identify Song")

        if identify:
            if not db.song_lengths:
                st.error("Database is empty. Add song files to the `songs/` directory.")
            else:
                with st.spinner("Fingerprinting and matching…"):
                    best, scores, fig_spec, fig_const, fig_hist = \
                        run_identification(uploaded.read(), uploaded.name)

                # ── Result banner ────────────────────────────────────────────
                if best:
                    st.markdown(
                        f'<div class="match-box">'
                        f'<span style="font-size:1.6rem">🎵</span> &nbsp;'
                        f'<strong style="font-size:1.3rem; color:#00ff88">'
                        f'Matched: {best}</strong>'
                        f'<br><span style="color:#88cc88; font-size:0.85rem">'
                        f'Score: {scores.get(best,0)} aligned hashes</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="no-match-box">'
                        '⚠️ <strong>No match found</strong> in the database.'
                        '</div>',
                        unsafe_allow_html=True
                    )

                st.markdown("---")

                # ── Intermediate steps ────────────────────────────────────────
                st.markdown("### Intermediate Steps")

                st.markdown('<p class="step-label">Step 1 — Spectrogram</p>',
                            unsafe_allow_html=True)
                st.image(fig_to_bytes(fig_spec), use_column_width=True)

                st.markdown('<p class="step-label">Step 2 — Constellation Map</p>',
                            unsafe_allow_html=True)
                st.image(fig_to_bytes(fig_const), use_column_width=True)

                st.markdown('<p class="step-label">Step 3 — Offset Histogram (match decision)</p>',
                            unsafe_allow_html=True)
                st.image(fig_to_bytes(fig_hist), use_column_width=True)

                # ── All scores ──────────────────────────────────────────────
                if scores:
                    st.markdown("### All Song Scores")
                    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
                    cols = st.columns(min(len(sorted_scores), 3))
                    for i, (sname, sc) in enumerate(sorted_scores):
                        with cols[i % len(cols)]:
                            delta = "✓ Best match" if sname == best else ""
                            st.metric(label=sname, value=sc, delta=delta)


# ─────────────────────────────────────────────────────────────
#  MODE 2 — Batch
# ─────────────────────────────────────────────────────────────
else:
    st.title("📂 Zapp tain America — Batch Identification")
    st.markdown(
        "Upload multiple query clips at once. Results are exported as "
        "`results.csv` with columns `filename, prediction`."
    )

    uploaded_files = st.file_uploader(
        "Upload query clips", type=["mp3", "wav"], accept_multiple_files=True
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) queued.**")
        run_batch = st.button("▶ Run Batch Identification")

        if run_batch:
            if not db.song_lengths:
                st.error("Database is empty. Add song files to the `songs/` directory.")
            else:
                rows = []
                progress = st.progress(0, text="Processing…")
                status_table = st.empty()

                for i, uf in enumerate(uploaded_files):
                    with st.spinner(f"Processing {uf.name}…"):
                        try:
                            best, scores, *_ = run_identification(uf.read(), uf.name)
                            prediction = best if best else "NO_MATCH"
                        except Exception as e:
                            prediction = "ERROR"
                        rows.append({"filename": uf.name, "prediction": prediction})

                    progress.progress((i + 1) / len(uploaded_files),
                                      text=f"Processed {i+1}/{len(uploaded_files)}")

                # Build results table
                import pandas as pd
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)

                # Download CSV (exact format required by assignment)
                csv_buf = io.StringIO()
                writer = csv.DictWriter(csv_buf, fieldnames=["filename", "prediction"])
                writer.writeheader()
                writer.writerows(rows)
                csv_bytes = csv_buf.getvalue().encode()

                st.download_button(
                    label="⬇ Download results.csv",
                    data=csv_bytes,
                    file_name="results.csv",
                    mime="text/csv",
                )
                progress.empty()
