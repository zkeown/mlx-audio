#!/usr/bin/env python3
"""Pre-compute spectrograms for E-GMD dataset.

This script generates .spec.npy files for all audio files in the E-GMD dataset,
dramatically speeding up training by avoiding on-the-fly spectrogram computation.

Usage:
    python -m mlx_audio.models.drumux.precompute_specs --data /path/to/e-gmd

    # With specific number of workers
    python -m mlx_audio.models.drumux.precompute_specs --data /path/to/e-gmd --workers 8

    # Force recompute existing files
    python -m mlx_audio.models.drumux.precompute_specs --data /path/to/e-gmd --force
"""

import argparse
import csv
from pathlib import Path

from mlx_audio.train import MelSpectrogramPreprocessor, run_precompute_cli

from .data import SpectrogramConfig


def get_audio_files(dataset_dir: Path) -> list[Path]:
    """Get all audio files from E-GMD dataset."""
    csv_path = dataset_dir / "e-gmd-v1.0.0.csv"

    if not csv_path.exists():
        subdirs = list(dataset_dir.glob("e-gmd-*"))
        if subdirs:
            csv_path = subdirs[0] / "e-gmd-v1.0.0.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"E-GMD CSV not found at {csv_path}")

    base_dir = csv_path.parent
    audio_files = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = base_dir / row["audio_filename"]
            if audio_path.exists():
                audio_files.append(audio_path)

    return audio_files


def main():
    parser = argparse.ArgumentParser(description="Pre-compute spectrograms for E-GMD dataset")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to E-GMD dataset directory",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if .spec.npy already exists",
    )

    args = parser.parse_args()

    # Get spectrogram config from DrumUX
    config = SpectrogramConfig()

    # Create preprocessor using framework utility
    preprocessor = MelSpectrogramPreprocessor(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
        power=2.0,
        normalize=True,
        suffix=".spec.npy",
    )

    # Get audio files
    print(f"Scanning {args.data}...")
    audio_files = get_audio_files(args.data)

    # Run pre-computation using framework CLI utility
    run_precompute_cli(
        input_files=audio_files,
        preprocessor=preprocessor,
        description="Pre-computing Spectrograms for E-GMD Dataset",
        num_workers=args.workers,
        force=args.force,
    )


if __name__ == "__main__":
    main()
