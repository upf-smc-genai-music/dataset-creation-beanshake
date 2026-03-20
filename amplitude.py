#!/usr/bin/env python3
"""
add_amplitude_envelope.py

Reads audio files from a folder, computes a smoothed amplitude envelope
at 75 samples/second, and adds it as a new column to the corresponding CSV files.

Usage:
    python add_amplitude_envelope.py --input_dir ./data [--output_dir ./output]
                                     [--column_name amplitude_envelope]
                                     [--smoothing_ms 40] [--normalize]

Dependencies:
    pip install numpy scipy soundfile pandas
"""

import argparse
import os
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, filtfilt
from pathlib import Path


# ─── Signal Processing ────────────────────────────────────────────────────────

def compute_amplitude_envelope(audio: np.ndarray, sample_rate: int,
                                target_fps: int = 75,
                                smoothing_ms: float = 40.0) -> np.ndarray:
    """
    Compute a smooth amplitude envelope from a waveform, resampled to target_fps.

    Steps:
      1. Mix to mono if stereo.
      2. Full-wave rectify (abs value).
      3. Low-pass filter to smooth rapid transients — cutoff tuned so that
         individual bean-hit peaks are captured but high-frequency noise is removed.
      4. Downsample to target_fps by averaging within each frame window.
      5. Optionally normalize to [0, 1].

    Args:
        audio:        Raw waveform samples (float, any channel count).
        sample_rate:  Native sample rate of the audio file (Hz).
        target_fps:   Output samples per second (default 75).
        smoothing_ms: Low-pass filter window in milliseconds. Larger = smoother
                      envelope; smaller = more transient detail. (default 40 ms)

    Returns:
        1-D numpy array of length == ceil(num_audio_samples / hop_size).
    """
    # 1. Mono mix
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # 2. Rectify
    rectified = np.abs(audio)

    # 3. Low-pass Butterworth filter
    #    cutoff = 1 / (2 * smoothing_ms) in Hz — preserves envelope shape
    #    while removing individual cycle ripple.
    cutoff_hz = 1000.0 / (2.0 * smoothing_ms)          # e.g. 40 ms → 12.5 Hz
    nyq = sample_rate / 2.0
    cutoff_norm = min(cutoff_hz / nyq, 0.99)            # must be < 1
    b, a = butter(4, cutoff_norm, btype='low')
    envelope_full = filtfilt(b, a, rectified)
    envelope_full = np.clip(envelope_full, 0, None)     # no negatives after filter

    # 4. Downsample: average samples within each target-fps frame
    hop = sample_rate / target_fps                      # samples per output frame
    n_frames = int(np.ceil(len(envelope_full) / hop))
    envelope_down = np.zeros(n_frames, dtype=np.float32)

    for i in range(n_frames):
        start = int(round(i * hop))
        end   = int(round((i + 1) * hop))
        end   = min(end, len(envelope_full))
        if start < end:
            envelope_down[i] = envelope_full[start:end].mean()

    return envelope_down


def normalize_envelope(envelope: np.ndarray) -> np.ndarray:
    """Scale envelope to [0, 1] using the file-level peak."""
    peak = envelope.max()
    if peak > 0:
        return envelope / peak
    return envelope


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
                 column_name: str, smoothing_ms: float, normalize: bool,
                 target_fps: int = 75) -> None:
    """Load one audio+CSV pair, add envelope column, save to output_dir."""

    # ── Load audio ──
    audio, sr = sf.read(str(audio_path), dtype='float32', always_2d=False)

    # ── Compute envelope ──
    envelope = compute_amplitude_envelope(audio, sr,
                                          target_fps=target_fps,
                                          smoothing_ms=smoothing_ms)
    if normalize:
        envelope = normalize_envelope(envelope)

    # ── Load CSV ──
    df = pd.read_csv(str(csv_path))
    n_rows = len(df)

    # ── Align lengths ──
    if len(envelope) > n_rows:
        # Trim extra envelope frames (audio slightly longer than CSV)
        envelope = envelope[:n_rows]
    elif len(envelope) < n_rows:
        # Pad with zeros (rare: CSV longer than audio)
        pad = np.zeros(n_rows - len(envelope), dtype=np.float32)
        envelope = np.concatenate([envelope, pad])
        print(f"  [warn] {audio_path.name}: envelope shorter than CSV "
              f"({len(envelope)} vs {n_rows}). Padded with zeros.")

    # ── Add column ──
    df[column_name] = envelope.astype(np.float32)

    # ── Save ──
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / csv_path.name
    df.to_csv(str(out_path), index=False)
    print(f"  ✓  {csv_path.name}  →  {out_path}  "
          f"(envelope min={envelope.min():.4f}, max={envelope.max():.4f})")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Add amplitude envelope column to RVQ/RNeNcodec CSV parameter files.")
    p.add_argument('--input_dir',    required=True,
                   help='Folder containing .wav (or other audio) + .csv pairs.')
    p.add_argument('--output_dir',   default=None,
                   help='Where to write updated CSVs. Defaults to input_dir/with_envelope/.')
    p.add_argument('--column_name',  default='amplitude_envelope',
                   help='Name of the new CSV column. (default: amplitude_envelope)')
    p.add_argument('--smoothing_ms', type=float, default=40.0,
                   help='Low-pass filter window in ms. Larger = smoother. (default: 40)')
    p.add_argument('--normalize',    action='store_true',
                   help='Normalize envelope to [0, 1] per file.')
    p.add_argument('--fps',          type=int, default=75,
                   help='Target parameter rate in samples/sec. (default: 75)')
    return p.parse_args()


def main():
    args = parse_args()
    input_dir  = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir \
                 else input_dir / f'with_envelope_{args.smoothing_ms}_ms'

    print(f"\nInput  : {input_dir}")
    print(f"Output : {output_dir}")
    print(f"Column : '{args.column_name}'  |  smoothing={args.smoothing_ms} ms  |  "
          f"normalize={args.normalize}  |  fps={args.fps}\n")

    pairs = list(find_pairs(input_dir))
    if not pairs:
        print("No matching audio+CSV pairs found. Exiting.")
        return

    for audio_path, csv_path in pairs:
        process_pair(audio_path, csv_path, output_dir,
                     column_name=args.column_name,
                     smoothing_ms=args.smoothing_ms,
                     normalize=args.normalize,
                     target_fps=args.fps)

    print(f"\nDone. {len(pairs)} file(s) processed.")


if __name__ == '__main__':
    main()