#!/usr/bin/env python3
"""Voice activity detection with Silero VAD.

This example demonstrates:
- Detecting speech segments in audio
- Adjusting detection sensitivity
- Extracting speech portions

Usage:
    python 06_detect_speech.py [audio_file]
"""

import sys
from pathlib import Path

import mlx_audio


def main():
    # Use provided audio file or a default
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = "sample_speech.wav"
        if not Path(audio_path).exists():
            print(f"No audio file found. Please provide an audio file:")
            print(f"  python {sys.argv[0]} path/to/audio.wav")
            print("\nOr run download_samples.py first to get sample files.")
            return

    print(f"Detecting speech in: {audio_path}")
    print("-" * 50)

    # Detect speech segments
    result = mlx_audio.detect_speech(audio_path)

    print(f"Found {len(result.segments)} speech segments")
    print(f"Total speech duration: {result.speech_duration:.1f}s")
    print(f"Speech ratio: {result.speech_ratio:.1%}")

    # Show segments
    print("\nSpeech segments:")
    for i, seg in enumerate(result.segments[:10]):  # First 10
        print(f"  {i + 1}. {seg.start:.2f}s - {seg.end:.2f}s ({seg.duration:.2f}s)")

    if len(result.segments) > 10:
        print(f"  ... and {len(result.segments) - 10} more")


def example_sensitivity():
    """Example adjusting detection sensitivity."""
    audio_path = "sample_speech.wav"

    # High sensitivity (catches more speech, may include noise)
    result_high = mlx_audio.detect_speech(
        audio_path,
        threshold=0.3,  # Lower = more sensitive
    )

    # Low sensitivity (stricter, misses quiet speech)
    result_low = mlx_audio.detect_speech(
        audio_path,
        threshold=0.7,  # Higher = less sensitive
    )

    print(f"High sensitivity: {len(result_high.segments)} segments")
    print(f"Low sensitivity: {len(result_low.segments)} segments")


def example_segment_merging():
    """Example controlling segment merging."""
    audio_path = "sample_speech.wav"

    # Merge nearby segments (for continuous speech)
    result = mlx_audio.detect_speech(
        audio_path,
        min_speech_duration=0.5,  # Ignore very short segments
        min_silence_duration=0.3,  # Merge if silence < 0.3s
    )

    print(f"Merged segments: {len(result.segments)}")


def example_preprocess_for_transcription():
    """Example using VAD to preprocess audio for transcription."""
    audio_path = "long_recording.wav"

    # Detect speech segments
    vad_result = mlx_audio.detect_speech(audio_path)

    print(f"Original audio: {vad_result.total_duration:.1f}s")
    print(f"Speech only: {vad_result.speech_duration:.1f}s")
    print(f"Speedup: {vad_result.total_duration / vad_result.speech_duration:.1f}x")

    # Transcribe only speech segments (faster)
    for i, seg in enumerate(vad_result.segments):
        print(f"\nTranscribing segment {i + 1}: {seg.start:.1f}s - {seg.end:.1f}s")
        # In practice, you'd extract the segment and transcribe it
        # result = mlx_audio.transcribe(segment_audio)


if __name__ == "__main__":
    main()
