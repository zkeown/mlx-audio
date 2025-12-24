#!/usr/bin/env python3
"""Music source separation with HTDemucs.

This example demonstrates:
- Separating music into stems (vocals, drums, bass, other)
- Saving individual stems
- Accessing stem audio data

Usage:
    python 02_separate.py [audio_file]
"""

import sys
from pathlib import Path

import mlx_audio


def main():
    # Use provided audio file or a default
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = "sample_music.mp3"
        if not Path(audio_path).exists():
            print(f"No audio file found. Please provide an audio file:")
            print(f"  python {sys.argv[0]} path/to/music.mp3")
            print("\nOr run download_samples.py first to get sample files.")
            return

    print(f"Separating: {audio_path}")
    print("-" * 50)

    # Separate into stems
    stems = mlx_audio.separate(audio_path)

    # Show available stems
    print(f"Available stems: {stems.stem_names}")

    # Access individual stems
    print(f"\nVocals shape: {stems.vocals.shape}")
    print(f"Drums shape: {stems.drums.shape}")
    print(f"Bass shape: {stems.bass.shape}")
    print(f"Other shape: {stems.other.shape}")

    # Save all stems to a directory
    output_dir = Path(audio_path).stem + "_stems"
    stems.save(output_dir)
    print(f"\nStems saved to: {output_dir}/")

    # Or save individual stems
    # stems.vocals.save("vocals.wav")
    # stems.drums.save("drums.wav")


def example_specific_stems():
    """Example requesting only specific stems."""
    audio_path = "sample_music.mp3"

    # Only extract vocals and drums (faster)
    stems = mlx_audio.separate(
        audio_path,
        stems=["vocals", "drums"],
    )

    # Only requested stems are available
    print(f"Vocals: {stems.vocals.shape}")
    print(f"Drums: {stems.drums.shape}")
    # stems.bass would raise AttributeError


def example_karaoke():
    """Create a karaoke version (music without vocals)."""
    audio_path = "sample_music.mp3"

    stems = mlx_audio.separate(audio_path)

    # Combine all stems except vocals for karaoke
    karaoke = stems.drums + stems.bass + stems.other
    karaoke.save("karaoke.wav")
    print("Karaoke version saved!")


if __name__ == "__main__":
    main()
