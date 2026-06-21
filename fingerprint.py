"""
fingerprint.py
--------------
Shazam-style audio fingerprinting engine.
Implements:
  - Spectrogram computation (STFT-based)
  - Constellation map extraction (local maxima)
  - Hash generation (paired peaks)
  - Database indexing
  - Query matching via offset histogram
"""

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from collections import defaultdict
import pickle, os, io

# ─────────────────────────────────────────────
#  STFT / Spectrogram parameters
# ─────────────────────────────────────────────
SR          = 22050   # resample rate
N_FFT       = 2048    # FFT window size  (~93 ms)
HOP_LENGTH  = 512     # hop (~23 ms → good time resolution)
N_MELS      = None    # use linear STFT, not mel

# Constellation parameters
NEIGHBORHOOD = (20, 20)   # (freq_bins, time_frames) local max window
MIN_AMPLITUDE_DB = -60    # ignore very quiet bins

# Hash / pairing parameters
FAN_VALUE   = 15      # how many peaks to pair with each anchor
TIME_DELTA_MIN = 1    # min time-frame gap between paired peaks
TIME_DELTA_MAX = 200  # max time-frame gap


# ─────────────────────────────────────────────
#  1. Spectrogram
# ─────────────────────────────────────────────

def compute_spectrogram(audio: np.ndarray, sr: int = SR,
                        n_fft: int = N_FFT,
                        hop_length: int = HOP_LENGTH):
    """
    Returns (S_db, freqs, times)
      S_db    – magnitude spectrogram in dB  (shape: freq_bins × time_frames)
      freqs   – frequency axis in Hz
      times   – time axis in seconds
    """
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S_db.shape[1]),
                                   sr=sr, hop_length=hop_length)
    return S_db, freqs, times


def load_audio(path: str, sr: int = SR, duration=None) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    return y


# ─────────────────────────────────────────────
#  2. Constellation (local maxima)
# ─────────────────────────────────────────────

def get_constellation(S_db: np.ndarray,
                      neighborhood: tuple = NEIGHBORHOOD,
                      min_amp_db: float = MIN_AMPLITUDE_DB):
    """
    Returns list of (freq_bin, time_frame) peak positions.
    A bin is a peak if it equals the maximum in its neighborhood
    and is above the minimum amplitude threshold.
    """
    local_max = maximum_filter(S_db, size=neighborhood)
    peaks = (S_db == local_max) & (S_db > min_amp_db)
    freq_idxs, time_idxs = np.where(peaks)
    # sort by time
    order = np.argsort(time_idxs)
    return list(zip(freq_idxs[order], time_idxs[order]))


# ─────────────────────────────────────────────
#  3. Hashing (paired peaks)
# ─────────────────────────────────────────────

def generate_hashes(constellation: list,
                    fan_value: int = FAN_VALUE,
                    dt_min: int = TIME_DELTA_MIN,
                    dt_max: int = TIME_DELTA_MAX):
    """
    For each anchor peak, pair it with up to fan_value future peaks.
    Hash = (freq1, freq2, delta_t)
    Yields (hash_tuple, anchor_time_frame)
    """
    hashes = []
    n = len(constellation)
    for i, (f1, t1) in enumerate(constellation):
        for j in range(i + 1, min(i + fan_value + 1, n)):
            f2, t2 = constellation[j]
            dt = t2 - t1
            if dt_min <= dt <= dt_max:
                hashes.append(((f1, f2, dt), t1))
    return hashes


def generate_single_peak_hashes(constellation: list):
    """
    Single-peak fingerprint: just (freq_bin, time_frame) pairs.
    Used for comparison experiment in Q3A.
    """
    return [((f,), t) for f, t in constellation]


# ─────────────────────────────────────────────
#  4. Database
# ─────────────────────────────────────────────

class FingerprintDB:
    """
    Stores hashes for all indexed songs.
    db[hash_tuple] = list of (song_name, anchor_time_frame)
    """
    def __init__(self):
        self.db = defaultdict(list)          # hash → [(song, t_anchor)]
        self.song_lengths = {}               # song_name → num_frames

    def index_song(self, song_name: str, audio: np.ndarray, sr: int = SR):
        S_db, _, _ = compute_spectrogram(audio, sr)
        constellation = get_constellation(S_db)
        hashes = generate_hashes(constellation)
        for h, t in hashes:
            self.db[h].append((song_name, t))
        self.song_lengths[song_name] = S_db.shape[1]
        return len(hashes), len(constellation)

    def index_from_file(self, path: str, sr: int = SR):
        song_name = os.path.splitext(os.path.basename(path))[0]
        audio = load_audio(path, sr=sr)
        return self.index_song(song_name, audio, sr)

    def match(self, query_audio: np.ndarray, sr: int = SR,
              use_single_peaks: bool = False):
        """
        Returns (best_match_name, offset_histogram_dict, all_scores)
          offset_histogram_dict: {song_name: {offset: count}}
          all_scores: {song_name: best_count}
        """
        S_db, _, _ = compute_spectrogram(query_audio, sr)
        constellation = get_constellation(S_db)

        if use_single_peaks:
            query_hashes = generate_single_peak_hashes(constellation)
        else:
            query_hashes = generate_hashes(constellation)

        # offset histogram: for each song, count how many hashes align
        # at the same time offset (query_t_anchor - db_t_anchor)
        offset_hist = defaultdict(lambda: defaultdict(int))

        for h, q_t in query_hashes:
            if h in self.db:
                for (song_name, db_t) in self.db[h]:
                    offset = q_t - db_t
                    offset_hist[song_name][offset] += 1

        # score = max count at any single offset
        scores = {}
        for song, hist in offset_hist.items():
            scores[song] = max(hist.values()) if hist else 0

        if not scores:
            return None, {}, {}

        best = max(scores, key=scores.get)
        return best, dict(offset_hist), scores

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'db': dict(self.db),
                         'song_lengths': self.song_lengths}, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.db = defaultdict(list, data['db'])
        self.song_lengths = data.get('song_lengths', {})


# ─────────────────────────────────────────────
#  5. Plotting helpers
# ─────────────────────────────────────────────

def plot_spectrogram(S_db, freqs, times, title="Spectrogram",
                     figsize=(12, 4), ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(S_db, aspect='auto', origin='lower',
                    extent=[times[0], times[-1], freqs[0], freqs[-1]],
                    cmap='magma', vmin=-80, vmax=0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    plt.colorbar(img, ax=ax, label="Amplitude (dB)")
    if standalone:
        plt.tight_layout()
        return fig
    return ax


def plot_constellation(S_db, freqs, times, constellation, title="Constellation",
                       figsize=(12, 4), ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(S_db, aspect='auto', origin='lower',
                    extent=[times[0], times[-1], freqs[0], freqs[-1]],
                    cmap='magma', vmin=-80, vmax=0, alpha=0.6)
    if constellation:
        f_idxs, t_idxs = zip(*constellation)
        peak_times = [times[min(t, len(times)-1)] for t in t_idxs]
        peak_freqs = [freqs[min(f, len(freqs)-1)] for f in f_idxs]
        ax.scatter(peak_times, peak_freqs, c='cyan', s=6,
                   linewidths=0.4, edgecolors='white', zorder=3,
                   label=f'{len(constellation)} peaks')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc='upper right')
    plt.colorbar(img, ax=ax, label="dB")
    if standalone:
        plt.tight_layout()
        return fig
    return ax


def plot_offset_histogram(offset_hist, song_name, title=None, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 3))
    hist = offset_hist.get(song_name, {})
    if hist:
        offsets = sorted(hist.keys())
        counts  = [hist[o] for o in offsets]
        ax.bar(offsets, counts, width=1, color='#00c8ff', edgecolor='none')
        best_offset = max(hist, key=hist.get)
        ax.axvline(best_offset, color='red', lw=1.5,
                   label=f'Peak offset = {best_offset}')
        ax.legend(fontsize=8)
    ax.set_xlabel("Time Offset (frames)")
    ax.set_ylabel("Matching Hashes")
    ax.set_title(title or f"Offset Histogram — {song_name}")
    if standalone:
        plt.tight_layout()
        return fig
    return ax


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf
