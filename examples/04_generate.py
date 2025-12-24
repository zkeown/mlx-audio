#!/usr/bin/env python3
"""Text-to-music generation with MusicGen.

This example demonstrates:
- Generating music from text descriptions
- Controlling generation parameters
- Saving generated audio

NOTE: MusicGen is licensed under CC-BY-NC-4.0 (non-commercial use only).
Commercial use requires a separate license from Meta.

Usage:
    python 04_generate.py [prompt]
"""

import sys

import mlx_audio

# Check license before proceeding
if not mlx_audio.is_commercial_safe("musicgen-medium"):
    print("NOTE: MusicGen is for non-commercial use only.")
    print("See: https://github.com/facebookresearch/audiocraft")
    print("-" * 50)


def main():
    # Use provided prompt or a default
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = "upbeat jazz piano solo, energetic and cheerful"

    print(f"Generating music for: '{prompt}'")
    print("-" * 50)

    # Generate music
    result = mlx_audio.generate(
        prompt,
        model="musicgen-medium",
        duration=10.0,  # seconds
    )

    print(f"Generated {result.duration:.1f}s of audio")
    print(f"Sample rate: {result.sample_rate} Hz")

    # Save to file
    output_path = "generated_music.wav"
    result.save(output_path)
    print(f"Saved to: {output_path}")


def example_with_options():
    """Example with various generation parameters."""
    prompt = "ambient electronic music, atmospheric and dreamy"

    result = mlx_audio.generate(
        prompt,
        model="musicgen-medium",
        duration=15.0,  # longer output
        temperature=0.8,  # more random (default 1.0)
        top_k=250,  # sampling parameter
        cfg_scale=3.0,  # classifier-free guidance
        seed=42,  # reproducible generation
    )

    result.save("ambient.wav")


def example_progress_callback():
    """Example with progress updates."""

    def on_progress(step: int, total: int):
        pct = step / total * 100
        bar = "=" * int(pct / 5) + ">" + " " * (20 - int(pct / 5))
        print(f"\rGenerating: [{bar}] {pct:.0f}%", end="", flush=True)

    result = mlx_audio.generate(
        "orchestral film score, epic and dramatic",
        duration=10.0,
        progress_callback=on_progress,
    )

    print("\nDone!")
    result.save("epic.wav")


def example_models():
    """Compare different MusicGen model sizes."""
    prompt = "acoustic guitar melody"

    models = [
        ("musicgen-small", "Fast, lower quality"),
        ("musicgen-medium", "Balanced (recommended)"),
        ("musicgen-large", "Slow, highest quality"),
    ]

    for model_name, description in models:
        print(f"\n{model_name}: {description}")
        result = mlx_audio.generate(prompt, model=model_name, duration=5.0)
        result.save(f"{model_name}_output.wav")


if __name__ == "__main__":
    main()
