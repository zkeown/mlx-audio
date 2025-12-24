#!/usr/bin/env python3
"""Text-to-speech synthesis with Parler-TTS.

This example demonstrates:
- Basic text-to-speech
- Voice customization via description
- Speed control
- Saving generated speech

Usage:
    python 05_speak.py [text]
"""

import sys

import mlx_audio


def main():
    # Use provided text or a default
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = "Hello! Welcome to mlx-audio. This is a demonstration of text to speech synthesis running natively on Apple Silicon."

    print(f"Generating speech for: '{text[:50]}...'")
    print("-" * 50)

    # Generate speech
    result = mlx_audio.speak(text)

    print(f"Generated {result.duration:.1f}s of speech")
    print(f"Sample rate: {result.sample_rate} Hz")

    # Save to file
    output_path = "speech.wav"
    result.save(output_path)
    print(f"Saved to: {output_path}")


def example_with_voice_description():
    """Example customizing voice characteristics."""
    text = "Welcome to the quarterly earnings report."

    # Describe the voice you want
    result = mlx_audio.speak(
        text,
        description="A professional male voice, clear and authoritative, "
        "speaking at a moderate pace with a slight British accent.",
    )

    result.save("professional_voice.wav")
    print("Generated with custom voice description")


def example_speed_control():
    """Example adjusting speech speed."""
    text = "This sentence will be spoken at different speeds."

    # Slow speech
    result = mlx_audio.speak(text, speed=0.75)
    result.save("slow_speech.wav")

    # Normal speech
    result = mlx_audio.speak(text, speed=1.0)
    result.save("normal_speech.wav")

    # Fast speech
    result = mlx_audio.speak(text, speed=1.25)
    result.save("fast_speech.wav")

    print("Generated three versions at different speeds")


def example_podcast_intro():
    """Example creating a podcast-style intro."""
    intro_text = """
    Welcome to Tech Talk, the podcast where we explore the latest in
    artificial intelligence and machine learning. I'm your host, and
    today we have a very special episode for you.
    """

    result = mlx_audio.speak(
        intro_text.strip(),
        description="A warm, engaging female voice with a friendly tone, "
        "speaking clearly and enthusiastically like a podcast host.",
        temperature=0.9,  # Slightly more expressive
    )

    result.save("podcast_intro.wav")
    print("Generated podcast intro")


def example_models():
    """Compare different Parler-TTS model sizes."""
    text = "The quick brown fox jumps over the lazy dog."

    models = [
        ("parler-tts-mini", "Fast, good quality"),
        ("parler-tts-large", "Slower, highest quality"),
    ]

    for model_name, description in models:
        print(f"\n{model_name}: {description}")
        result = mlx_audio.speak(text, model=model_name)
        result.save(f"{model_name}_output.wav")


if __name__ == "__main__":
    main()
