#!/usr/bin/env python3
"""
extract_features.py

Reads audio files from a folder, computes a full suite of rhythmic and
intensity features at 75 samples/second, and adds them as new columns to
the corresponding CSV files.

New columns added (all at target_fps resolution):
  - amplitude_envelope   : smoothed full-wave rectified amplitude
  - rms_db               : frame-level RMS energy in decibels
  - spectral_flux        : L1 spectral flux (positive difference only)
  - onset_strength       : librosa onset strength envelope
  - ioi_s                : inter-onset interval in seconds (interpolated)
  - shake_rate_bpm       : instantaneous shake rate in beats per minute

Usage:
    python extract_features.py --input_dir ./data [--output_dir ./output]
                               [--smoothing_ms 40] [--normalize]
                               [--onset_delta 0.07] [--fps 75]

Dependencies:
    pip install numpy scipy soundfile pandas librosa
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt
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


# ── 1. Amplitude Envelope ─────────────────────────────────────────────────────

def compute_amplitude_envelope(audio: np.ndarray, sample_rate: int,
                                target_fps: int = 75,
                                smoothing_ms: float = 40.0) -> np.ndarray:
    """
    Smoothed amplitude envelope resampled to target_fps.

    Steps:
      1. Mono mix → full-wave rectify.
      2. 4th-order Butterworth low-pass (cutoff = 1 / (2 * smoothing_ms)).
      3. Downsample by frame-averaging to target_fps.
    """
    mono = to_mono(audio)
    rectified = np.abs(mono)

    cutoff_hz   = 1000.0 / (2.0 * smoothing_ms)       # 40 ms → 12.5 Hz
    nyq         = sample_rate / 2.0
    cutoff_norm = min(cutoff_hz / nyq, 0.99)
    b, a        = butter(4, cutoff_norm, btype='low')
    env_full    = filtfilt(b, a, rectified)
    env_full    = np.clip(env_full, 0, None)

    hop      = sample_rate / target_fps
    n_frames = int(np.ceil(len(env_full) / hop))
    env_down = np.zeros(n_frames, dtype=np.float32)

    for i in range(n_frames):
        start = int(round(i * hop))
        end   = min(int(round((i + 1) * hop)), len(env_full))
        if start < end:
            env_down[i] = env_full[start:end].mean()

    return env_down


# ── 2. RMS in dB ──────────────────────────────────────────────────────────────

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


# ── 3. Spectral Flux ──────────────────────────────────────────────────────────

def compute_spectral_flux(audio: np.ndarray, sample_rate: int,
                          target_fps: int = 75) -> np.ndarray:
    """
    Half-wave rectified L1 spectral flux (positive spectral changes only).

    Captures the sudden broadband energy burst of each bean-shake onset
    without being fooled by decay.  One value per target-fps frame.
    """
    mono       = to_mono(audio)
    hop_length = int(round(sample_rate / target_fps))

    # STFT magnitude spectrum
    S = np.abs(librosa.stft(mono, hop_length=hop_length))  # (freq_bins, n_frames)

    # Positive first-difference along the time axis, summed over frequency bins
    diff = np.diff(S, axis=1)
    flux = np.sum(np.maximum(diff, 0.0), axis=0).astype(np.float32)

    # Prepend a zero so shape matches n_frames (same convention as librosa onset_strength)
    flux = np.concatenate([[0.0], flux])

    return flux


# ── 4. Onset Strength ─────────────────────────────────────────────────────────

def compute_onset_strength(audio: np.ndarray, sample_rate: int,
                            target_fps: int = 75) -> np.ndarray:
    """
    Librosa onset strength envelope at target_fps.

    Uses a mel-scaled spectrogram internally; more perceptually weighted
    than raw spectral flux and robust to the diffuse spectral texture of
    rattling beans.
    """
    mono       = to_mono(audio)
    hop_length = int(round(sample_rate / target_fps))

    onset_env = librosa.onset.onset_strength(
        y=mono, sr=sample_rate, hop_length=hop_length
    ).astype(np.float32)

    return onset_env


# ── 5. IOI (seconds) + Shake Rate (BPM) ──────────────────────────────────────

def compute_ioi_and_bpm(audio: np.ndarray, sample_rate: int,
                        target_fps: int = 75,
                        onset_delta: float = 0.07) -> tuple[np.ndarray, np.ndarray]:
    """
    Inter-Onset Interval (IOI) in seconds and shake rate in BPM,
    both interpolated to a continuous target_fps signal.

    Strategy:
      1. Detect onset frames with librosa (adaptive threshold).
      2. Convert frame indices → times → IOI series (np.diff of times).
      3. Linearly interpolate IOI back onto the full frame grid.
         Between the first detected onset and the start of audio, the IOI
         is held at the first measured value (forward fill from left).
         Same for the tail.
      4. BPM = 60 / IOI.

    Args:
        onset_delta:  Threshold delta for onset_detect — higher = fewer,
                      more confident onsets (0.07 is a good default for
                      percussive transients; raise to ~0.15 for dense shaking).

    Returns:
        (ioi_s, shake_rate_bpm): each a 1-D float32 array of length n_frames.
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
              "IOI and BPM will be zero for this file.")
        zeros = np.zeros(n_frames, dtype=np.float32)
        return zeros, zeros

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
        fill_value=(anchor_ioi[0], anchor_ioi[-1]),       # flat extrapolation
    )
    all_frames = np.arange(n_frames, dtype=float)
    ioi_s      = ioi_interp_fn(all_frames).astype(np.float32)
    ioi_s      = np.clip(ioi_s, 1e-3, None)               # prevent division by zero

    shake_rate_bpm = (60.0 / ioi_s).astype(np.float32)

    return ioi_s, shake_rate_bpm


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
                 smoothing_ms: float, normalize: bool,
                 onset_delta: float, target_fps: int = 75) -> None:
    """
    Load one audio+CSV pair, compute all features, append columns, save.

    Columns written:
        amplitude_envelope  – smoothed rectified envelope
        rms_db              – frame RMS in dB  (range −80 … 0)
        spectral_flux       – half-rectified L1 spectral flux
        onset_strength      – librosa mel-onset strength
        ioi_s               – inter-onset interval (seconds)
        shake_rate_bpm      – instantaneous shake rate (BPM)
    """
    print(f"\n  Processing: {audio_path.name}")

    # ── Load audio ──────────────────────────────────────────────────────────
    audio, sr = sf.read(str(audio_path), dtype='float32', always_2d=False)

    # ── Compute all features ─────────────────────────────────────────────────
    amp_env    = compute_amplitude_envelope(audio, sr, target_fps, smoothing_ms)
    rms_db     = compute_rms_db(audio, sr, target_fps)
    flux       = compute_spectral_flux(audio, sr, target_fps)
    onset_str  = compute_onset_strength(audio, sr, target_fps)
    ioi_s, bpm = compute_ioi_and_bpm(audio, sr, target_fps, onset_delta)

    # ── Optionally normalise continuous features to [0, 1] ──────────────────
    if normalize:
        amp_env   = normalize_feature(amp_env)
        # rms_db: shift from [-80, 0] → [0, 1] via linear map (preserves meaning)
        # rms_db    = (rms_db + 80.0) / 80.0
        flux      = normalize_feature(flux)
        onset_str = normalize_feature(onset_str)
        ioi_s     = normalize_feature(ioi_s)
        # bpm       = normalize_feature(bpm)

    # ── Load CSV and determine canonical frame count ─────────────────────────
    df       = pd.read_csv(str(csv_path))
    n_rows   = len(df)

    # ── Align every feature to n_rows ────────────────────────────────────────
    features = {
        'amplitude_envelope': amp_env,
        'rms_db':             rms_db,
        'spectral_flux':      flux,
        'onset_strength':     onset_str,
        'ioi_s':              ioi_s,
        'shake_rate_bpm':     bpm,
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
    print(f"       amplitude_envelope : min={features['amplitude_envelope'].min():.4f}  "
          f"max={features['amplitude_envelope'].max():.4f}")
    print(f"       rms_db             : min={features['rms_db'].min():.4f}  "
          f"max={features['rms_db'].max():.4f}")
    print(f"       spectral_flux      : min={features['spectral_flux'].min():.4f}  "
          f"max={features['spectral_flux'].max():.4f}")
    print(f"       onset_strength     : min={features['onset_strength'].min():.4f}  "
          f"max={features['onset_strength'].max():.4f}")
    print(f"       ioi_s              : min={features['ioi_s'].min():.4f}  "
          f"max={features['ioi_s'].max():.4f}")
    print(f"       shake_rate_bpm     : min={features['shake_rate_bpm'].min():.2f}  "
          f"max={features['shake_rate_bpm'].max():.2f}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Extract rhythmic and intensity features from audio files and "
            "append them to matching CSV parameter files at 75 fps.\n\n"
            "Columns added: amplitude_envelope, rms_db, spectral_flux, "
            "onset_strength, ioi_s, shake_rate_bpm."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--input_dir',    required=True,
                   help='Folder containing .wav (or other audio) + .csv pairs.')
    p.add_argument('--output_dir',   default=None,
                   help='Where to write updated CSVs. '
                        'Defaults to input_dir/with_features/.')
    p.add_argument('--smoothing_ms', type=float, default=40.0,
                   help='Low-pass filter window (ms) for amplitude envelope. '
                        'Larger = smoother. (default: 40)')
    p.add_argument('--onset_delta',  type=float, default=0.07,
                   help='Onset detection threshold delta. '
                        'Raise (e.g. 0.15) for very dense/fast shaking to '
                        'avoid double-triggers. (default: 0.07)')
    p.add_argument('--normalize',    action='store_true',
                   help='Normalize all features to [0, 1] per file.')
    p.add_argument('--fps',          type=int, default=75,
                   help='Target parameter rate in samples/sec. (default: 75)')
    return p.parse_args()


def main():
    args = parse_args()
    input_dir  = Path(args.input_dir).resolve()
    output_dir = (Path(args.output_dir).resolve() if args.output_dir
                  else input_dir / 'with_features')

    print(f"\nInput       : {input_dir}")
    print(f"Output      : {output_dir}")
    print(f"FPS         : {args.fps}")
    print(f"Smoothing   : {args.smoothing_ms} ms")
    print(f"Onset delta : {args.onset_delta}")
    print(f"Normalize   : {args.normalize}\n")

    pairs = list(find_pairs(input_dir))
    if not pairs:
        print("No matching audio+CSV pairs found. Exiting.")
        return

    for audio_path, csv_path in pairs:
        process_pair(
            audio_path, csv_path, output_dir,
            smoothing_ms=args.smoothing_ms,
            normalize=args.normalize,
            onset_delta=args.onset_delta,
            target_fps=args.fps,
        )

    print(f"\nDone. {len(pairs)} file(s) processed.")


if __name__ == '__main__':
    main()