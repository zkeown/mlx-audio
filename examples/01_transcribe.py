#!/usr/bin/env python3
"""Speech-to-text transcription with Whisper.

This example demonstrates:
- Basic transcription
- Language detection
- Subtitle generation (SRT format)
- Word-level timestamps

Usage:
    python 01_transcribe.py [audio_file]
"""

import sys
from pathlib import Path

import mlx_audio


def main():
    # Use provided audio file or a default
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        # Create a simple test or use existing file
        audio_path = "sample_speech.wav"
        if not Path(audio_path).exists():
            print(f"No audio file found. Please provide an audio file:")
            print(f"  python {sys.argv[0]} path/to/audio.wav")
            print("\nOr run download_samples.py first to get sample files.")
            return

    print(f"Transcribing: {audio_path}")
    print("-" * 50)

    # Basic transcription
    result = mlx_audio.transcribe(audio_path)

    print(f"Text: {result.text}")
    print(f"Language: {result.language}")
    print(f"Duration: {result.duration:.1f}s")

    # Show segments with timestamps
    if result.segments:
        print("\nSegments:")
        for seg in result.segments[:5]:  # Show first 5
            print(f"  [{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")

    # Save as subtitles
    output_path = Path(audio_path).stem + ".srt"
    result.save(output_path, format="srt")
    print(f"\nSubtitles saved to: {output_path}")


def example_with_options():
    """Example showing additional transcription options."""
    audio_path = "sample_speech.wav"

    # Transcribe with specific options
    result = mlx_audio.transcribe(
        audio_path,
        model="whisper-large-v3-turbo",  # Fast and accurate
        language="en",  # Skip language detection
        task="transcribe",  # or "translate" to English
        word_timestamps=True,  # Get word-level timing
    )

    print("Word-level timestamps:")
    for word in result.words[:10]:
        print(f"  [{word.start:.2f}s] {word.text}")


def example_translate():
    """Example showing translation to English."""
    audio_path = "french_speech.wav"

    # Translate non-English audio to English
    result = mlx_audio.transcribe(
        audio_path,
        task="translate",  # Translate to English
    )

    print(f"Original language: {result.language}")
    print(f"English translation: {result.text}")


if __name__ == "__main__":
    main()
