#!/usr/bin/env python3
"""Generate MUSDB18-HQ fixtures for Swift integration and parity testing.

This script creates fixtures from MUSDB18-HQ tracks that can be used to:
1. Test Swift HTDemucs separation quality
2. Verify Python↔Swift parity on real audio

Usage:
    python generate_musdb18_fixtures.py \
        --musdb18-root /path/to/MUSDB18HQ \
        --output-dir swift/Tests/Fixtures/Integration/musdb18 \
        --tracks 5

Output structure:
    output_dir/
    ├── manifest.json           # Track list and metadata
    ├── track_0/
    │   ├── metadata.json       # Track info
    │   ├── mixture.safetensors # Input mixture
    │   ├── stems.safetensors   # Ground truth stems
    │   └── python_output.safetensors  # Python separation output
    ├── track_1/
    │   └── ...
    └── ...
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf


def load_track(track_path: Path, duration: float | None = None) -> dict[str, np.ndarray]:
    """Load a MUSDB18-HQ track with all stems.

    Args:
        track_path: Path to track directory
        duration: Optional duration in seconds (from start)

    Returns:
        Dictionary with mixture and stems as [channels, samples] arrays
    """
    stems = {}
    stem_names = ["mixture", "drums", "bass", "other", "vocals"]

    for stem in stem_names:
        wav_path = track_path / f"{stem}.wav"
        audio, sr = sf.read(wav_path, dtype="float32")
        assert sr == 44100, f"Expected 44100 Hz, got {sr}"

        # Transpose to [channels, samples]
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        else:
            audio = audio.T

        # Truncate if duration specified
        if duration is not None:
            max_samples = int(duration * sr)
            audio = audio[:, :max_samples]

        stems[stem] = audio

    return stems


def separate_with_python(mixture: np.ndarray) -> np.ndarray:
    """Run HTDemucs separation using Python implementation.

    Args:
        mixture: Audio array [channels, samples]

    Returns:
        Separated sources [4, channels, samples]
    """
    from mlx_audio.models.demucs import HTDemucs
    from mlx_audio.models.demucs.inference import apply_model

    # Load model
    model = HTDemucs.from_pretrained("htdemucs_ft")

    # Convert to MLX
    mix_mx = mx.array(mixture)

    # Separate
    sources = apply_model(model, mix_mx, segment=6.0, overlap=0.25)

    return np.array(sources)


def save_safetensors(path: Path, arrays: dict[str, np.ndarray]):
    """Save arrays to safetensors format."""
    try:
        from safetensors.numpy import save_file

        save_file(arrays, str(path))
    except ImportError:
        # Fallback to MLX safetensors
        mx_arrays = {k: mx.array(v) for k, v in arrays.items()}
        mx.save_safetensors(str(path), mx_arrays)


def generate_track_fixtures(
    track_path: Path,
    output_dir: Path,
    track_index: int,
    duration: float = 30.0,
    generate_python_output: bool = True,
) -> dict:
    """Generate fixtures for a single track.

    Args:
        track_path: Path to MUSDB18-HQ track directory
        output_dir: Output directory for fixtures
        track_index: Index for naming
        duration: Duration to extract (seconds)
        generate_python_output: Whether to run Python separation

    Returns:
        Metadata dictionary for the track
    """
    track_name = track_path.name
    track_output = output_dir / f"track_{track_index}"
    track_output.mkdir(parents=True, exist_ok=True)

    print(f"  Loading track: {track_name}")

    # Load track
    data = load_track(track_path, duration=duration)
    sample_rate = 44100
    actual_duration = data["mixture"].shape[1] / sample_rate

    # Save mixture
    print(f"    Saving mixture...")
    save_safetensors(
        track_output / "mixture.safetensors",
        {"mixture": data["mixture"].astype(np.float32)},
    )

    # Save ground truth stems
    print(f"    Saving stems...")
    save_safetensors(
        track_output / "stems.safetensors",
        {
            "drums": data["drums"].astype(np.float32),
            "bass": data["bass"].astype(np.float32),
            "other": data["other"].astype(np.float32),
            "vocals": data["vocals"].astype(np.float32),
        },
    )

    # Generate Python separation output
    if generate_python_output:
        print(f"    Running Python separation...")
        sources = separate_with_python(data["mixture"])
        save_safetensors(
            track_output / "python_output.safetensors",
            {
                "drums": sources[0].astype(np.float32),
                "bass": sources[1].astype(np.float32),
                "other": sources[2].astype(np.float32),
                "vocals": sources[3].astype(np.float32),
            },
        )

    # Save metadata
    metadata = {
        "track_name": track_name,
        "track_index": track_index,
        "duration_seconds": actual_duration,
        "sample_rate": sample_rate,
        "channels": 2,
        "samples": data["mixture"].shape[1],
        "has_python_output": generate_python_output,
    }

    with open(track_output / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def get_quick_tracks(musdb18_root: Path) -> list[Path]:
    """Get 5 representative tracks for quick testing."""
    test_dir = musdb18_root / "test"
    all_tracks = sorted([d for d in test_dir.iterdir() if d.is_dir()])

    # Preferred tracks for variety
    preferred = [
        "Al James - Schoolboy Facination",
        "AM Contra - Heart Peripheral",
        "Angels In Amplifiers - I'm Alright",
        "Arise - Run Run Run",
        "BKS - Bulldozer",
    ]

    selected = []
    for name in preferred:
        path = test_dir / name
        if path.exists():
            selected.append(path)
        if len(selected) >= 5:
            break

    # Fill with remaining tracks if needed
    for track in all_tracks:
        if track not in selected:
            selected.append(track)
        if len(selected) >= 5:
            break

    return selected[:5]


def main():
    parser = argparse.ArgumentParser(
        description="Generate MUSDB18-HQ fixtures for Swift testing"
    )
    parser.add_argument(
        "--musdb18-root",
        type=Path,
        required=True,
        help="Path to MUSDB18-HQ root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for fixtures",
    )
    parser.add_argument(
        "--tracks",
        type=int,
        default=5,
        help="Number of tracks to process (default: 5)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Duration in seconds per track (default: 30)",
    )
    parser.add_argument(
        "--no-python-output",
        action="store_true",
        help="Skip generating Python separation outputs",
    )
    parser.add_argument(
        "--split",
        choices=["test", "train"],
        default="test",
        help="Which split to use (default: test)",
    )

    args = parser.parse_args()

    # Validate paths
    if not args.musdb18_root.exists():
        print(f"Error: MUSDB18-HQ root not found: {args.musdb18_root}")
        sys.exit(1)

    split_dir = args.musdb18_root / args.split
    if not split_dir.exists():
        print(f"Error: Split directory not found: {split_dir}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Get tracks
    if args.tracks <= 5:
        tracks = get_quick_tracks(args.musdb18_root)[:args.tracks]
    else:
        all_tracks = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        tracks = all_tracks[:args.tracks]

    print(f"Generating fixtures for {len(tracks)} tracks...")
    print(f"Output directory: {args.output_dir}")
    print(f"Duration per track: {args.duration}s")
    print()

    # Generate fixtures
    manifest = {
        "dataset": "MUSDB18-HQ",
        "split": args.split,
        "num_tracks": len(tracks),
        "duration_per_track": args.duration,
        "model": "htdemucs_ft",
        "tracks": [],
    }

    for i, track_path in enumerate(tracks):
        print(f"\n[{i + 1}/{len(tracks)}] {track_path.name}")
        metadata = generate_track_fixtures(
            track_path,
            args.output_dir,
            i,
            duration=args.duration,
            generate_python_output=not args.no_python_output,
        )
        manifest["tracks"].append(metadata)

    # Save manifest
    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Generated fixtures for {len(tracks)} tracks")
    print(f"Output directory: {args.output_dir}")
    print(f"Manifest: {manifest_path}")
    print()
    print("To run Swift parity tests:")
    print(f"  INTEGRATION_FIXTURES={args.output_dir} swift test --filter IntegrationTests")


if __name__ == "__main__":
    main()
