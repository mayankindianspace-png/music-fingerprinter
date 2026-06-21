"""
app.py — Q3B: Zapp tain America (self-contained, no external imports)
Compatible with Streamlit 1.58+ / Python 3.14
"""

# ── Standard imports ──────────────────────────────────────────────────────────
import streamlit as st
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, os, csv, tempfile, pickle
from pathlib import Path
from collections import defaultdict
from scipy.ndimage import maximum_filter

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
SONGS_DIR = BASE_DIR / "songs"
DB_PATH   = BASE_DIR / "fingerprint_db.pkl"

# ── Audio / STFT parameters ───────────────────────────────────────────────────
SR          = 22050
N_FFT       = 2048
HOP_LENGTH  = 512
NEIGHBORHOOD    = (20, 20)
MIN_AMP_DB      = -60
FAN_VALUE       = 15
TIME_DELTA_MIN  = 1
TIME_DELTA_MAX  = 200

# ─────────────────────────────────────────────────────────────────────────────
#  Core fingerprinting functions
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path, sr=SR, duration=None):
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    return y

def compute_spectrogram(audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH):
    S     = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    S_db  = librosa.amplitude_to_db(S, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length)
    return S_db, freqs, times

def get_constellation(S_db, neighborhood=NEIGHBORHOOD, min_amp_db=MIN_AMP_DB):
    local_max = maximum_filter(S_db, size=neighborhood)
    peaks     = (S_db == local_max) & (S_db > min_amp_db)
    f_idx, t_idx = np.where(peaks)
    order = np.argsort(t_idx)
    return list(zip(f_idx[order], t_idx[order]))

def generate_hashes(constellation, fan_value=FAN_VALUE,
                    dt_min=TIME_DELTA_MIN, dt_max=TIME_DELTA_MAX):
    hashes, n = [], len(constellation)
    for i, (f1, t1) in enumerate(constellation):
        for j in range(i + 1, min(i + fan_value + 1, n)):
            f2, t2 = constellation[j]
            dt = t2 - t1
            if dt_min <= dt <= dt_max:
                hashes.append(((f1, f2, dt), t1))
    return hashes

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

# ─────────────────────────────────────────────────────────────────────────────
#  Database class
# ─────────────────────────────────────────────────────────────────────────────

class FingerprintDB:
    def __init__(self):
        self.db           = defaultdict(list)
        self.song_lengths = {}

    def index_song(self, song_name, audio, sr=SR):
        S_db, _, _ = compute_spectrogram(audio, sr)
        constellation = get_constellation(S_db)
        for h, t in generate_hashes(constellation):
            self.db[h].append((song_name, t))
        self.song_lengths[song_name] = S_db.shape[1]

    def index_from_file(self, path, sr=SR):
        name  = Path(path).stem
        audio = load_audio(path, sr=sr)
        self.index_song(name, audio, sr)

    def match(self, query_audio, sr=SR):
        S_db, _, _    = compute_spectrogram(query_audio, sr)
        constellation = get_constellation(S_db)
        query_hashes  = generate_hashes(constellation)
        offset_hist   = defaultdict(lambda: defaultdict(int))
        for h, q_t in query_hashes:
            if h in self.db:
                for (song_name, db_t) in self.db[h]:
                    offset_hist[song_name][q_t - db_t] += 1
        scores = {s: max(hist.values()) for s, hist in offset_hist.items() if hist}
        best   = max(scores, key=scores.get) if scores else None
        return best, dict(offset_hist), scores

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'db': dict(self.db), 'song_lengths': self.song_lengths}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.db           = defaultdict(list, data['db'])
        self.song_lengths = data.get('song_lengths', {})

# ─────────────────────────────────────────────────────────────────────────────
#  Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dark_style():
    plt.rcParams.update({
        'figure.facecolor': '#0d0d0d', 'axes.facecolor': '#12122a',
        'axes.edgecolor':   '#333',    'axes.labelcolor': '#ccc',
        'xtick.color':      '#999',    'ytick.color':     '#999',
        'text.color':       '#ddd',
    })

def plot_spectrogram(S_db, freqs, times, title="Spectrogram", ax=None):
    standalone = ax is None
    if standalone:
        _dark_style(); fig, ax = plt.subplots(figsize=(9, 3.5))
    img = ax.imshow(S_db, aspect='auto', origin='lower',
                    extent=[times[0], times[-1], freqs[0], freqs[-1]],
                    cmap='magma', vmin=-80, vmax=0)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)"); ax.set_title(title)
    plt.colorbar(img, ax=ax, label="dB")
    if standalone:
        plt.tight_layout(); return plt.gcf()

def plot_constellation(S_db, freqs, times, constellation, title="Constellation", ax=None):
    standalone = ax is None
    if standalone:
        _dark_style(); fig, ax = plt.subplots(figsize=(9, 3.5))
    img = ax.imshow(S_db, aspect='auto', origin='lower',
                    extent=[times[0], times[-1], freqs[0], freqs[-1]],
                    cmap='magma', vmin=-80, vmax=0, alpha=0.6)
    if constellation:
        fi, ti = zip(*constellation)
        pt = [times[min(t, len(times)-1)] for t in ti]
        pf = [freqs[min(f, len(freqs)-1)] for f in fi]
        ax.scatter(pt, pf, c='cyan', s=6, edgecolors='white',
                   linewidths=0.4, zorder=3, label=f'{len(constellation)} peaks')
        ax.legend(fontsize=8, loc='upper right')
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)"); ax.set_title(title)
    plt.colorbar(img, ax=ax, label="dB")
    if standalone:
        plt.tight_layout(); return plt.gcf()

def plot_offset_histogram(offset_hist, song_name, title=None, ax=None):
    standalone = ax is None
    if standalone:
        _dark_style(); fig, ax = plt.subplots(figsize=(9, 3))
    hist = offset_hist.get(song_name, {})
    if hist:
        offsets = sorted(hist.keys())
        ax.bar(offsets, [hist[o] for o in offsets], width=1,
               color='#00c8ff', edgecolor='none')
        best_off = max(hist, key=hist.get)
        ax.axvline(best_off, color='red', lw=1.5, label=f'Peak = {best_off}')
        ax.legend(fontsize=8)
    ax.set_xlabel("Time Offset (frames)"); ax.set_ylabel("Matching Hashes")
    ax.set_title(title or f"Offset Histogram — {song_name}")
    if standalone:
        plt.tight_layout(); return plt.gcf()

# ─────────────────────────────────────────────────────────────────────────────
#  Page config & CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zapp tain America • Music Fingerprinter",
    page_icon="🎵", layout="wide",
)
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#0a0a14; color:#e0e0f0; }
[data-testid="stSidebar"]          { background:#10101e; border-right:1px solid #2a2a4a; }
h1  { color:#00c8ff; letter-spacing:1px; }
h2,h3 { color:#a0c4ff; }
.stButton>button {
    background:linear-gradient(135deg,#00c8ff 0%,#5b5bff 100%);
    color:#fff; border:none; border-radius:8px;
    font-weight:600; padding:0.5rem 1.5rem;
}
.match-box    { background:#0d2a0d; border:2px solid #00ff88;
                border-radius:10px; padding:1.2rem 1.6rem; margin-top:0.8rem; }
.no-match-box { background:#2a0d0d; border:2px solid #ff4444;
                border-radius:10px; padding:1.2rem 1.6rem; margin-top:0.8rem; }
.step-label   { font-size:0.78rem; color:#888;
                text-transform:uppercase; letter-spacing:1.5px; margin-bottom:0.2rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Load / build database (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Indexing song database…")
def get_database():
    db = FingerprintDB()
    if DB_PATH.exists():
        try:
            db.load(str(DB_PATH))
            if db.song_lengths:
                return db
        except Exception:
            pass
    if SONGS_DIR.exists():
        for p in sorted(SONGS_DIR.glob("*.mp3")) + sorted(SONGS_DIR.glob("*.wav")):
            try:
                db.index_from_file(str(p))
            except Exception as e:
                st.warning(f"Could not index {p.name}: {e}")
        try:
            db.save(str(DB_PATH))
        except Exception:
            pass
    return db

db = get_database()

# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────
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
    st.caption(f"DB: {len(db.db):,} hashes · {len(db.song_lengths)} songs")

# ─────────────────────────────────────────────────────────────────────────────
#  Identify helper
# ─────────────────────────────────────────────────────────────────────────────
def run_identification(audio_bytes, filename):
    suffix = Path(filename).suffix or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes); tmp_path = tmp.name
    try:
        audio = load_audio(tmp_path)
    finally:
        try: os.unlink(tmp_path)
        except Exception: pass

    _dark_style()
    S_db, freqs, times = compute_spectrogram(audio)
    constellation      = get_constellation(S_db)
    best, offset_hist, scores = db.match(audio)

    fig_spec,  ax1 = plt.subplots(figsize=(9, 3.5))
    plot_spectrogram(S_db, freqs, times, title="Spectrogram of Query Clip", ax=ax1)
    plt.tight_layout()

    fig_const, ax2 = plt.subplots(figsize=(9, 3.5))
    plot_constellation(S_db, freqs, times, constellation,
                       title=f"Constellation Map ({len(constellation)} peaks)", ax=ax2)
    plt.tight_layout()

    fig_hist,  ax3 = plt.subplots(figsize=(9, 3))
    if best:
        plot_offset_histogram(offset_hist, best,
                              title=f"Offset Histogram — Best: '{best}'", ax=ax3)
    else:
        ax3.text(0.5, 0.5, "No match found", ha='center', va='center',
                 transform=ax3.transAxes, color='#ff4444', fontsize=13)
        ax3.set_title("Offset Histogram")
    plt.tight_layout()

    return best, scores, fig_spec, fig_const, fig_hist

# ─────────────────────────────────────────────────────────────────────────────
#  Single Clip mode
# ─────────────────────────────────────────────────────────────────────────────
if mode == "Single Clip":
    st.title("🎵 Zapp tain America — Single Clip Identifier")
    st.markdown("Upload a short audio clip (MP3 or WAV) and the system will identify it.")

    uploaded = st.file_uploader("Upload query clip", type=["mp3", "wav"])
    if uploaded:
        st.audio(uploaded)
        if st.button("🔍 Identify Song"):
            if not db.song_lengths:
                st.error("Database is empty — no songs found in `songs/` folder.")
            else:
                with st.spinner("Fingerprinting and matching…"):
                    best, scores, fig_spec, fig_const, fig_hist = \
                        run_identification(uploaded.read(), uploaded.name)

                if best:
                    st.markdown(
                        f'<div class="match-box">🎵 &nbsp;'
                        f'<strong style="font-size:1.3rem;color:#00ff88">Matched: {best}</strong>'
                        f'<br><span style="color:#88cc88;font-size:0.85rem">'
                        f'Score: {scores.get(best,0)} aligned hashes</span></div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div class="no-match-box">⚠️ <strong>No match found.</strong></div>',
                        unsafe_allow_html=True)

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
                    cols = st.columns(min(len(scores), 3))
                    for i, (sname, sc) in enumerate(sorted(scores.items(), key=lambda x:-x[1])):
                        with cols[i % len(cols)]:
                            st.metric(sname, sc,
                                      delta="✓ Best match" if sname == best else "")

# ─────────────────────────────────────────────────────────────────────────────
#  Batch mode
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.title("📂 Zapp tain America — Batch Identification")
    st.markdown("Upload multiple clips → download `results.csv` (filename, prediction).")

    uploaded_files = st.file_uploader("Upload query clips", type=["mp3","wav"],
                                      accept_multiple_files=True)
    if uploaded_files and st.button("▶ Run Batch Identification"):
        if not db.song_lengths:
            st.error("Database is empty.")
        else:
            rows, progress = [], st.progress(0, text="Processing…")
            for i, uf in enumerate(uploaded_files):
                try:
                    best, *_ = run_identification(uf.read(), uf.name)
                    pred = best if best else "NO_MATCH"
                except Exception as e:
                    pred = "ERROR"; st.warning(f"{uf.name}: {e}")
                rows.append({"filename": uf.name, "prediction": pred})
                progress.progress((i+1)/len(uploaded_files), text=f"{i+1}/{len(uploaded_files)} done")

            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            buf = io.StringIO()
            w   = csv.DictWriter(buf, fieldnames=["filename","prediction"])
            w.writeheader(); w.writerows(rows)
            st.download_button("⬇ Download results.csv",
                               buf.getvalue().encode(), "results.csv", "text/csv")
            progress.empty()
