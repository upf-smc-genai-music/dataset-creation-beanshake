#!/usr/bin/env python3
"""
extract_features.py

Reads audio files from a folder, computes two physically independent
control features at 75 samples/second, and adds them as new columns to
the corresponding CSV files.

These two features map directly to the two user-facing model toggles:

  - rms_db           : frame-level RMS energy in decibels  → intensity toggle
  - shake_rate_bpm   : instantaneous shake rate in BPM     → rhythm toggle

Both are derived from independent signal paths so they remain decorrelated
even when the training data has incidental covariance between rate and loudness.

Usage:
    python extract_features.py --input_dir ./data [--output_dir ./output]
                               [--normalize] [--smoothing_ms 20]
                               [--peak_prominence 0.1] [--min_ioi_ms 100]
                               [--fps 75]

Dependencies:
    pip install numpy scipy soundfile pandas librosa
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import interp1d
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)   # suppress librosa warnings


# ─── Signal Processing ────────────────────────────────────────────────────────

def to_mono(audio: np.ndarray) -> np.ndarray:
    """Mix down to mono if multi-channel."""
    if audio.ndim > 1:
        return audio.mean(axis=1)
    return audio


def _align(arr: np.ndarray, n_frames: int, name: str) -> np.ndarray:
    """Trim or zero-pad a 1-D array to exactly n_frames."""
    if len(arr) > n_frames:
        return arr[:n_frames]
    if len(arr) < n_frames:
        print(f"    [warn] {name}: {len(arr)} frames computed, "
              f"padding to {n_frames} with zeros.")
        return np.concatenate([arr, np.zeros(n_frames - len(arr), dtype=np.float32)])
    return arr


def normalize_feature(arr: np.ndarray) -> np.ndarray:
    """Min-max scale to [0, 1]; safe against all-zero arrays."""
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        return (arr - lo) / (hi - lo)
    return np.zeros_like(arr)


# ── 1. RMS in dB ──────────────────────────────────────────────────────────────

def compute_rms_db(audio: np.ndarray, sample_rate: int,
                   target_fps: int = 75) -> np.ndarray:
    """
    Frame-level RMS energy in dB, resampled to target_fps.
    Silence floor is clamped at -80 dB.
    """
    mono       = to_mono(audio)
    hop_length = int(round(sample_rate / target_fps))

    rms    = librosa.feature.rms(y=mono, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    rms_db = np.clip(rms_db, -80.0, 0.0).astype(np.float32)

    return rms_db


# ── 2. Amplitude Envelope (for peak detection) ────────────────────────────────

def compute_envelope(mono: np.ndarray, sample_rate: int,
                     smoothing_ms: float = 20.0) -> np.ndarray:
    """
    Full-wave rectified amplitude envelope, smoothed with a low-pass
    Butterworth filter.  Returned at the native sample rate — NOT
    downsampled — so that find_peaks has full resolution to work with.

    Args:
        mono:         Mono waveform at native sample rate.
        sample_rate:  Native sample rate (Hz).
        smoothing_ms: Low-pass cutoff window in ms.  Smaller values
                      preserve sharper transient peaks (default 20 ms).
                      Raise to 40+ ms for very sparse/slow shaking.

    Returns:
        env: 1-D float32 array, same length as mono, range [0, 1].
    """
    rectified   = np.abs(mono)
    cutoff_hz   = 1000.0 / (2.0 * smoothing_ms)
    nyq         = sample_rate / 2.0
    cutoff_norm = min(cutoff_hz / nyq, 0.99)
    b, a        = butter(4, cutoff_norm, btype='low')
    env         = filtfilt(b, a, rectified)
    env         = np.clip(env, 0, None).astype(np.float32)

    # Normalise to [0, 1] so prominence threshold is scale-independent
    peak = env.max()
    if peak > 0:
        env /= peak
    return env


# ── 3. Shake Rate (BPM) via envelope peaks ────────────────────────────────────

def compute_shake_rate_bpm(audio: np.ndarray, sample_rate: int,
                           target_fps: int = 75,
                           smoothing_ms: float = 20.0,
                           peak_prominence: float = 0.1,
                           min_ioi_ms: float = 100.0) -> np.ndarray:
    """
    Instantaneous shake rate in BPM derived from peaks in the amplitude
    envelope, interpolated to a continuous target_fps signal.

    Strategy:
      1. Build a smoothed amplitude envelope at native sample rate.
      2. Find peaks in the envelope using scipy.signal.find_peaks with a
         prominence threshold and a minimum inter-peak distance.
         These peaks correspond directly to individual shake events.
      3. Compute IOI (seconds) between consecutive peaks.
      4. Linearly interpolate IOI back onto the full frame grid at
         target_fps, with flat extrapolation at both edges.
      5. BPM = 60 / IOI.

    Args:
        smoothing_ms:     Low-pass window for envelope (ms).  Smaller =
                          sharper peaks, more sensitive to fast shaking.
        peak_prominence:  Minimum prominence for a peak to count as a shake,
                          expressed as a fraction of the normalised envelope
                          [0, 1].  Raise to ignore weak rattles; lower to
                          catch soft shakes. (default 0.1)
        min_ioi_ms:       Minimum allowed gap between two accepted peaks (ms).
                          Acts as a hard floor on BPM = 60 000 / min_ioi_ms.
                          Default 100 ms → max detectable rate = 600 BPM.
                          Raise (e.g. 150 ms) to suppress double-triggers on
                          a single shake. (default 100)

    Returns:
        shake_rate_bpm: 1-D float32 array of length n_frames.
    """
    mono       = to_mono(audio)
    hop_length = int(round(sample_rate / target_fps))

    # ── Build envelope at native sample rate ──────────────────────────────
    env = compute_envelope(mono, sample_rate, smoothing_ms)

    # ── Find peaks ────────────────────────────────────────────────────────
    min_distance_samples = int(round(min_ioi_ms * sample_rate / 1000.0))
    peak_samples, _ = find_peaks(
        env,
        prominence=peak_prominence,
        distance=min_distance_samples,
    )

    n_frames = int(np.ceil(len(mono) / hop_length))

    if len(peak_samples) < 2:
        print("    [warn] Fewer than 2 envelope peaks detected — "
              "try lowering --peak_prominence or --min_ioi_ms. "
              "shake_rate_bpm will be zero for this file.")
        return np.zeros(n_frames, dtype=np.float32)

    print(f"    [info] {len(peak_samples)} peaks detected.")

    # ── IOI in seconds between consecutive peaks ──────────────────────────
    peak_times = peak_samples / sample_rate          # convert samples → seconds
    ioi_values = np.diff(peak_times)                 # (n_peaks - 1,)

    # Anchor each IOI at the frame index of the earlier peak
    anchor_frames = (peak_samples[:-1] / hop_length).astype(float)
    anchor_ioi    = ioi_values.astype(float)

    # ── Interpolate onto target_fps frame grid ────────────────────────────
    ioi_interp_fn = interp1d(
        anchor_frames, anchor_ioi,
        kind='linear',
        bounds_error=False,
        fill_value=(anchor_ioi[0], anchor_ioi[-1]),  # flat extrapolation at edges
    )
    ioi_s = ioi_interp_fn(np.arange(n_frames, dtype=float)).astype(np.float32)
    ioi_s = np.clip(ioi_s, 1e-3, None)               # guard against division by zero

    return (60.0 / ioi_s).astype(np.float32)


# ─── File Matching ─────────────────────────────────────────────────────────────

AUDIO_EXTENSIONS = {'.wav', '.flac', '.ogg', '.aiff', '.aif', '.mp3'}


def find_pairs(input_dir: Path):
    """
    Yield (audio_path, csv_path) pairs where both share the same stem.
    Reports any audio files without a matching CSV (and vice-versa).
    """
    audio_files = {p.stem: p for p in input_dir.iterdir()
                   if p.suffix.lower() in AUDIO_EXTENSIONS}
    csv_files   = {p.stem: p for p in input_dir.iterdir()
                   if p.suffix.lower() == '.csv'}

    matched   = set(audio_files) & set(csv_files)
    unmatched = (set(audio_files) - matched) | (set(csv_files) - matched)

    if unmatched:
        print(f"  [warn] No pair found for: {', '.join(sorted(unmatched))}")

    for stem in sorted(matched):
        yield audio_files[stem], csv_files[stem]


# ─── Core Processing ───────────────────────────────────────────────────────────

def process_pair(audio_path: Path, csv_path: Path, output_dir: Path,
                 normalize: bool, smoothing_ms: float,
                 peak_prominence: float, min_ioi_ms: float,
                 target_fps: int = 75) -> None:
    """
    Load one audio+CSV pair, compute both control features, append columns, save.

    Columns written:
        rms_db           – frame RMS in dB  (range −80 … 0)  → intensity toggle
        shake_rate_bpm   – instantaneous shake rate in BPM    → rhythm toggle
    """
    print(f"\n  Processing: {audio_path.name}")

    # ── Load audio ──────────────────────────────────────────────────────────
    audio, sr = sf.read(str(audio_path), dtype='float32', always_2d=False)

    # ── Compute features ─────────────────────────────────────────────────────
    rms_db = compute_rms_db(audio, sr, target_fps)
    bpm    = compute_shake_rate_bpm(
                audio, sr, target_fps,
                smoothing_ms=smoothing_ms,
                peak_prominence=peak_prominence,
                min_ioi_ms=min_ioi_ms,
             )

    # ── Optionally normalise to [0, 1] ───────────────────────────────────────
    if normalize:
        # rms_db: linear map from [−80, 0] → [0, 1] (preserves perceptual scale)
        rms_db = (rms_db + 80.0) / 80.0
        bpm    = normalize_feature(bpm)

    # ── Load CSV and determine canonical frame count ─────────────────────────
    df     = pd.read_csv(str(csv_path))
    n_rows = len(df)

    # ── Align both features to n_rows ────────────────────────────────────────
    features = {
        'rms_db':         rms_db,
        'shake_rate_bpm': bpm,
    }
    for col, arr in features.items():
        features[col] = _align(arr.astype(np.float32), n_rows, col)

    # ── Append columns ────────────────────────────────────────────────────────
    for col, arr in features.items():
        df[col] = arr

    # ── Save ──────────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / csv_path.name
    df.to_csv(str(out_path), index=False)

    # ── Summary print ─────────────────────────────────────────────────────────
    print(f"  ✓  {csv_path.name}  →  {out_path}")
    print(f"       rms_db         : min={features['rms_db'].min():.4f}  "
          f"max={features['rms_db'].max():.4f}")
    print(f"       shake_rate_bpm : min={features['shake_rate_bpm'].min():.2f}  "
          f"max={features['shake_rate_bpm'].max():.2f}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Extract two independent control features from audio files and\n"
            "append them to matching CSV parameter files at 75 fps.\n\n"
            "Columns added:\n"
            "  rms_db           – intensity toggle  (dB, or [0,1] with --normalize)\n"
            "  shake_rate_bpm   – rhythm toggle     (BPM, or [0,1] with --normalize)\n\n"
            "Shake rate is computed from peaks in the smoothed amplitude envelope.\n"
            "Tune --peak_prominence and --min_ioi_ms if detection is over/under-triggering."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--input_dir',        required=True,
                   help='Folder containing .wav (or other audio) + .csv pairs.')
    p.add_argument('--output_dir',       default=None,
                   help='Where to write updated CSVs. '
                        'Defaults to input_dir/with_features/.')
    p.add_argument('--smoothing_ms',     type=float, default=20.0,
                   help='Envelope low-pass window in ms. Smaller = sharper peaks. '
                        '(default: 20)')
    p.add_argument('--peak_prominence',  type=float, default=0.1,
                   help='Min peak prominence as fraction of normalised envelope [0,1]. '
                        'Lower to catch soft shakes; raise to ignore weak rattles. '
                        '(default: 0.1)')
    p.add_argument('--min_ioi_ms',       type=float, default=100.0,
                   help='Minimum gap between peaks in ms — suppresses double-triggers. '
                        '100 ms → max detectable rate = 600 BPM. '
                        'Raise (e.g. 150) for slower shaking. (default: 100)')
    p.add_argument('--normalize',        action='store_true',
                   help='Normalize both features to [0, 1] per file.')
    p.add_argument('--fps',              type=int, default=75,
                   help='Target parameter rate in samples/sec. (default: 75)')
    return p.parse_args()


def main():
    args = parse_args()
    input_dir  = Path(args.input_dir).resolve()
    output_dir = (Path(args.output_dir).resolve() if args.output_dir
                  else input_dir / f'csv_{args.smoothing_ms}ms_norm_{args.normalize}_smoothing_ms_{args.smoothing_ms}_min_ioi_ms_{args.min_ioi_ms}')

    print(f"\nInput           : {input_dir}")
    print(f"Output          : {output_dir}")
    print(f"FPS             : {args.fps}")
    print(f"Smoothing       : {args.smoothing_ms} ms")
    print(f"Peak prominence : {args.peak_prominence}")
    print(f"Min IOI         : {args.min_ioi_ms} ms")
    print(f"Normalize       : {args.normalize}\n")

    pairs = list(find_pairs(input_dir))
    if not pairs:
        print("No matching audio+CSV pairs found. Exiting.")
        return

    for audio_path, csv_path in pairs:
        process_pair(
            audio_path, csv_path, output_dir,
            normalize=args.normalize,
            smoothing_ms=args.smoothing_ms,
            peak_prominence=args.peak_prominence,
            min_ioi_ms=args.min_ioi_ms,
            target_fps=args.fps,
        )

    print(f"\nDone. {len(pairs)} file(s) processed.")


if __name__ == '__main__':
    main()