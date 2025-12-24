#!/usr/bin/env python3
"""Speaker diarization with ECAPA-TDNN.

This example demonstrates:
- Identifying who spoke when
- Automatic speaker counting
- Combining with transcription

Usage:
    python 08_diarize.py [audio_file]
"""

import sys
from pathlib import Path

import mlx_audio


def main():
    # Use provided audio file or a default
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = "meeting.wav"
        if not Path(audio_path).exists():
            print(f"No audio file found. Please provide an audio file:")
            print(f"  python {sys.argv[0]} path/to/multi_speaker.wav")
            print("\nOr run download_samples.py first to get sample files.")
            return

    print(f"Diarizing: {audio_path}")
    print("-" * 50)

    # Detect speakers
    result = mlx_audio.diarize(audio_path)

    print(f"Detected {result.num_speakers} speakers")
    print(f"Total segments: {len(result.segments)}")

    # Show speaker turns
    print("\nSpeaker turns:")
    for seg in result.segments[:15]:  # First 15
        print(f"  {seg.speaker}: {seg.start:.1f}s - {seg.end:.1f}s ({seg.duration:.1f}s)")

    if len(result.segments) > 15:
        print(f"  ... and {len(result.segments) - 15} more")

    # Show speaking time per speaker
    print("\nSpeaking time by speaker:")
    for speaker, duration in result.speaking_time.items():
        pct = duration / result.total_duration * 100
        bar = "=" * int(pct / 2)
        print(f"  {speaker}: {duration:.1f}s ({pct:.0f}%) {bar}")


def example_known_speakers():
    """Example with known number of speakers."""
    audio_path = "interview.wav"

    # Provide known speaker count for better accuracy
    result = mlx_audio.diarize(
        audio_path,
        num_speakers=2,  # Interview = 2 speakers
    )

    print(f"Detected speakers: {result.speaker_ids}")


def example_with_limits():
    """Example constraining speaker count."""
    audio_path = "meeting.wav"

    # Constrain auto-detection range
    result = mlx_audio.diarize(
        audio_path,
        min_speakers=2,
        max_speakers=6,
    )

    print(f"Auto-detected {result.num_speakers} speakers")


def example_with_transcription():
    """Example combining diarization with transcription."""
    audio_path = "meeting.wav"

    # First, transcribe
    transcript = mlx_audio.transcribe(audio_path)

    # Then, diarize with transcript for speaker-labeled text
    result = mlx_audio.diarize(audio_path, transcription=transcript)

    # Now segments include text
    print("Speaker-labeled transcript:")
    for seg in result.segments[:10]:
        if hasattr(seg, "text") and seg.text:
            print(f"  [{seg.speaker}] {seg.text}")


def example_meeting_summary():
    """Example creating a meeting summary."""
    audio_path = "meeting.wav"

    # Diarize
    result = mlx_audio.diarize(audio_path)

    print("Meeting Summary")
    print("=" * 40)
    print(f"Duration: {result.total_duration:.0f} seconds")
    print(f"Participants: {result.num_speakers}")
    print()

    # Show participation stats
    print("Participation:")
    total_speech = sum(result.speaking_time.values())
    for speaker in sorted(result.speaking_time.keys()):
        duration = result.speaking_time[speaker]
        pct = duration / total_speech * 100
        print(f"  {speaker}: {pct:.0f}% ({duration:.0f}s)")


if __name__ == "__main__":
    main()
