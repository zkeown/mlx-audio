#!/usr/bin/env python3
"""Audio embeddings and zero-shot classification with CLAP.

This example demonstrates:
- Computing audio embeddings
- Computing text embeddings
- Zero-shot audio classification
- Audio-text similarity search

Usage:
    python 03_embed.py [audio_file]
"""

import sys
from pathlib import Path

import mlx_audio


def main():
    # Use provided audio file or a default
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = "sample_sound.wav"
        if not Path(audio_path).exists():
            print(f"No audio file found. Please provide an audio file:")
            print(f"  python {sys.argv[0]} path/to/audio.wav")
            print("\nOr run download_samples.py first to get sample files.")
            return

    print(f"Embedding: {audio_path}")
    print("-" * 50)

    # Zero-shot classification
    labels = [
        "dog barking",
        "cat meowing",
        "bird singing",
        "person speaking",
        "music playing",
        "car engine",
        "rain falling",
    ]

    result = mlx_audio.embed(
        audio=audio_path,
        text=labels,
        return_similarity=True,
    )

    # Get the best match
    best = result.best_match()
    print(f"Best match: {best}")

    # Show all similarities
    print("\nAll similarities:")
    for label, score in result.ranked_matches():
        bar = "=" * int(score * 20)
        print(f"  {label:20s} {score:.3f} {bar}")


def example_audio_embedding():
    """Example computing audio embeddings for similarity search."""
    audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]

    # Compute embeddings for multiple audio files
    embeddings = []
    for path in audio_files:
        result = mlx_audio.embed(audio=path)
        embeddings.append(result.audio_embeds)

    print(f"Computed {len(embeddings)} audio embeddings")
    print(f"Embedding shape: {embeddings[0].shape}")

    # You can use these for similarity search, clustering, etc.


def example_text_embedding():
    """Example computing text embeddings."""
    descriptions = [
        "a dog barking loudly",
        "soft piano music",
        "rain on a rooftop",
        "crowd cheering at a sports event",
    ]

    result = mlx_audio.embed(text=descriptions)
    print(f"Text embeddings shape: {result.text_embeds.shape}")


def example_classification():
    """Example using the classify convenience function."""
    audio_path = "sample_sound.wav"

    result = mlx_audio.classify(
        audio_path,
        labels=["speech", "music", "noise", "silence"],
    )

    print(f"Predicted: {result.predicted_class}")
    print(f"Confidence: {result.confidence:.1%}")


if __name__ == "__main__":
    main()
