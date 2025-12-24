#!/usr/bin/env python3
"""Download sample audio files for running examples.

This script downloads small audio samples that can be used with the
example scripts in this directory.
"""

import urllib.request
from pathlib import Path


def download_file(url: str, filename: str) -> bool:
    """Download a file if it doesn't exist."""
    path = Path(filename)
    if path.exists():
        print(f"  {filename} already exists, skipping")
        return True

    try:
        print(f"  Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"  Saved: {filename}")
        return True
    except Exception as e:
        print(f"  Failed to download {filename}: {e}")
        return False


def main():
    print("Downloading sample audio files...")
    print("-" * 50)

    # Sample files from public domain sources
    samples = [
        # LibriSpeech sample (public domain)
        (
            "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav",
            "sample_speech.wav",
        ),
    ]

    success = 0
    for url, filename in samples:
        if download_file(url, filename):
            success += 1

    print("-" * 50)
    print(f"Downloaded {success}/{len(samples)} files")

    print("\nNote: Some examples require additional audio files.")
    print("You can use your own audio files by passing them as arguments:")
    print("  python 01_transcribe.py your_audio.wav")

    # Create placeholder instructions for files we can't auto-download
    print("\nFor best results, provide your own:")
    print("  - sample_music.mp3   (any music file, for separation)")
    print("  - noisy_speech.wav   (speech with background noise, for enhancement)")
    print("  - meeting.wav        (multi-speaker recording, for diarization)")


if __name__ == "__main__":
    main()
