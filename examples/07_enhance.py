#!/usr/bin/env python3
"""Audio enhancement and denoising with DeepFilterNet.

This example demonstrates:
- Removing background noise from speech
- Comparing before/after quality
- Different enhancement methods

Usage:
    python 07_enhance.py [audio_file]
"""

import sys
from pathlib import Path

import mlx_audio


def main():
    # Use provided audio file or a default
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = "noisy_speech.wav"
        if not Path(audio_path).exists():
            print(f"No audio file found. Please provide an audio file:")
            print(f"  python {sys.argv[0]} path/to/noisy_audio.wav")
            print("\nOr run download_samples.py first to get sample files.")
            return

    print(f"Enhancing: {audio_path}")
    print("-" * 50)

    # Enhance audio (remove noise)
    result = mlx_audio.enhance(audio_path)

    print(f"Enhanced {result.duration:.1f}s of audio")
    print(f"Sample rate: {result.sample_rate} Hz")

    # Save enhanced audio
    output_path = Path(audio_path).stem + "_enhanced.wav"
    result.save(output_path)
    print(f"Saved to: {output_path}")


def example_compare_before_after():
    """Example comparing original and enhanced audio."""
    audio_path = "noisy_speech.wav"

    # Keep original for comparison
    result = mlx_audio.enhance(audio_path, keep_original=True)

    # Access both versions
    original, enhanced = result.before_after

    # Save both for comparison
    original.save("original.wav")
    enhanced.save("enhanced.wav")

    print("Saved original.wav and enhanced.wav for comparison")


def example_spectral_method():
    """Example using spectral gating (no model download)."""
    audio_path = "noisy_speech.wav"

    # Spectral gating works offline, no model needed
    result = mlx_audio.enhance(
        audio_path,
        method="spectral",  # Instead of "neural"
    )

    result.save("spectral_enhanced.wav")
    print("Enhanced with spectral gating (no model required)")


def example_podcast_cleanup():
    """Example cleaning up podcast audio."""
    audio_path = "podcast_recording.wav"

    # DeepFilterNet is optimized for speech
    result = mlx_audio.enhance(
        audio_path,
        model="deepfilternet2",
    )

    result.save("podcast_clean.wav")
    print("Cleaned podcast audio")


def example_batch_processing():
    """Example processing multiple files."""
    audio_files = ["recording1.wav", "recording2.wav", "recording3.wav"]

    for path in audio_files:
        if not Path(path).exists():
            continue

        print(f"Processing: {path}")
        result = mlx_audio.enhance(path)

        output_path = Path(path).stem + "_clean.wav"
        result.save(output_path)
        print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
