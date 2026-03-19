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
                               [--normalize] [--onset_delta 0.07] [--fps 75]

Dependencies:
    pip install numpy soundfile pandas librosa
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
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


# ── 2. Shake Rate (BPM) ───────────────────────────────────────────────────────

def compute_shake_rate_bpm(audio: np.ndarray, sample_rate: int,
                           target_fps: int = 75,
                           onset_delta: float = 0.07) -> np.ndarray:
    """
    Instantaneous shake rate in BPM, interpolated to a continuous target_fps signal.

    Strategy:
      1. Detect onset frames with librosa (adaptive threshold).
      2. Convert frame indices → times → IOI series (np.diff of times).
      3. Linearly interpolate IOI back onto the full frame grid with flat
         extrapolation at both edges (holds the nearest measured value).
      4. BPM = 60 / IOI.

    Args:
        onset_delta:  Threshold delta for onset_detect — higher = fewer,
                      more confident onsets (0.07 is a good default for
                      percussive transients; raise to ~0.15 for dense shaking).

    Returns:
        shake_rate_bpm: 1-D float32 array of length n_frames.
    """
    mono       = to_mono(audio)
    hop_length = int(round(sample_rate / target_fps))

    onset_env = librosa.onset.onset_strength(
        y=mono, sr=sample_rate, hop_length=hop_length
    )
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sample_rate,
        hop_length=hop_length,
        units='frames',
        delta=onset_delta,
        normalize=True,
    )

    n_frames = len(onset_env)

    # Need at least 2 onsets to compute IOI
    if len(onset_frames) < 2:
        print("    [warn] Fewer than 2 onsets detected — "
              "shake_rate_bpm will be zero for this file.")
        return np.zeros(n_frames, dtype=np.float32)

    onset_times = librosa.frames_to_time(
        onset_frames, sr=sample_rate, hop_length=hop_length
    )
    ioi_values = np.diff(onset_times)                     # (n_onsets - 1,)

    # Anchor IOI values at the *start* frame of each inter-onset segment
    anchor_frames = onset_frames[:-1].astype(float)
    anchor_ioi    = ioi_values.astype(float)

    # Edge-clamp: extrapolate flat to the first and last frame
    ioi_interp_fn = interp1d(
        anchor_frames, anchor_ioi,
        kind='linear',
        bounds_error=False,
        fill_value=(anchor_ioi[0], anchor_ioi[-1]),
    )
    ioi_s = ioi_interp_fn(np.arange(n_frames, dtype=float)).astype(np.float32)
    ioi_s = np.clip(ioi_s, 1e-3, None)                    # prevent division by zero

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
                 normalize: bool, onset_delta: float,
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
    bpm    = compute_shake_rate_bpm(audio, sr, target_fps, onset_delta)

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
            "Extract two independent control features from audio files and "
            "append them to matching CSV parameter files at 75 fps.\n\n"
            "Columns added: rms_db (intensity toggle), shake_rate_bpm (rhythm toggle)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--input_dir',    required=True,
                   help='Folder containing .wav (or other audio) + .csv pairs.')
    p.add_argument('--output_dir',   default=None,
                   help='Where to write updated CSVs. '
                        'Defaults to input_dir/with_features/.')
    p.add_argument('--onset_delta',  type=float, default=0.07,
                   help='Onset detection threshold delta. '
                        'Raise (e.g. 0.15) for very dense/fast shaking to '
                        'avoid double-triggers. (default: 0.07)')
    p.add_argument('--normalize',    action='store_true',
                   help='Normalize both features to [0, 1] per file.')
    p.add_argument('--fps',          type=int, default=75,
                   help='Target parameter rate in samples/sec. (default: 75)')
    return p.parse_args()


def main():
    args = parse_args()
    input_dir  = Path(args.input_dir).resolve()
    output_dir = (Path(args.output_dir).resolve() if args.output_dir
                  else input_dir / f'csv_norm_{args.normalize}_onset_delta_{args.onset_delta}')

    print(f"\nInput       : {input_dir}")
    print(f"Output      : {output_dir}")
    print(f"FPS         : {args.fps}")
    print(f"Onset delta : {args.onset_delta}")
    print(f"Normalize   : {args.normalize}\n")

    pairs = list(find_pairs(input_dir))
    if not pairs:
        print("No matching audio+CSV pairs found. Exiting.")
        return

    for audio_path, csv_path in pairs:
        process_pair(
            audio_path, csv_path, output_dir,
            normalize=args.normalize,
            onset_delta=args.onset_delta,
            target_fps=args.fps,
        )

    print(f"\nDone. {len(pairs)} file(s) processed.")


if __name__ == '__main__':
    main()