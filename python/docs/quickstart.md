# Quickstart

This guide provides detailed examples for all 10 high-level API functions in mlx-audio.

## Speech Transcription

Convert speech to text using Whisper models.

```python
import mlx_audio as ma

# Basic transcription
result = ma.transcribe("speech.wav")
print(result.text)

# Specify language (faster if known)
result = ma.transcribe("audio.wav", language="en")

# Translate to English
result = ma.transcribe("french.wav", task="translate")

# Save as subtitles
result.save("subtitles.srt", format="srt")

# Word-level timestamps
result = ma.transcribe("speech.wav", word_timestamps=True)
for word in result.words:
    print(f"{word.word}: {word.start:.2f}s - {word.end:.2f}s")
```

**Available models:** `whisper-large-v3-turbo` (default), `whisper-large-v3`, `whisper-large-v2`, `whisper-medium`, `whisper-small`, `whisper-base`, `whisper-tiny`

## Music Stem Separation

Separate audio into vocals, drums, bass, and other stems.

```python
import mlx_audio as ma

# Separate all stems
result = ma.separate("song.mp3")

# Access individual stems
print(result.vocals.shape)   # Vocal track
print(result.drums.shape)    # Drum track
print(result.bass.shape)     # Bass track
print(result.other.shape)    # Other instruments

# Save all stems
result.save("output/")  # Creates vocals.wav, drums.wav, etc.

# Save specific stems
result.vocals.save("vocals_only.wav")

# Request only specific stems
result = ma.separate("song.mp3", stems=["vocals", "drums"])
```

**Available models:** `htdemucs_ft` (default), `htdemucs`

## Music Generation

Generate audio from text descriptions using MusicGen.

```python
import mlx_audio as ma

# Basic generation
result = ma.generate("jazz piano, upbeat mood")
result.save("jazz.wav")

# Longer duration (max 30s)
result = ma.generate("ambient electronic music", duration=20.0)

# Control generation parameters
result = ma.generate(
    "rock guitar solo",
    duration=10.0,
    temperature=0.9,     # More randomness
    top_k=250,           # Top-k sampling
    cfg_scale=4.0,       # Stronger prompt adherence
    seed=42,             # Reproducible output
)

# Progress callback for long generations
def on_progress(step, total):
    print(f"Generating: {step}/{total}")

result = ma.generate(
    "orchestral music",
    duration=30.0,
    progress_callback=on_progress,
)
```

**Available models:** `musicgen-medium` (default), `musicgen-small`, `musicgen-large`, `musicgen-melody`

## Text-to-Speech

Convert text to natural speech using Parler-TTS.

```python
import mlx_audio as ma

# Basic synthesis
result = ma.speak("Hello, welcome to mlx-audio!")
result.save("greeting.wav")

# Voice customization via description
result = ma.speak(
    "Welcome to the presentation.",
    description="A professional male voice, clear and authoritative",
)

# Adjust speech speed
result = ma.speak("This is slower speech.", speed=0.8)
result = ma.speak("This is faster speech.", speed=1.3)

# Reproducible output
result = ma.speak(
    "Consistent voice output",
    seed=42,
    temperature=0.7,
)
```

**Available models:** `parler-tts-mini` (default), `parler-tts-large`

## Audio Embeddings

Compute audio and text embeddings using CLAP for similarity search and zero-shot classification.

```python
import mlx_audio as ma

# Audio embedding
result = ma.embed(audio="dog_bark.wav")
print(result.audio_embeds.shape)  # [1, 512]

# Text embedding
result = ma.embed(text="a dog barking loudly")
print(result.text_embeds.shape)  # [1, 512]

# Multiple text embeddings
result = ma.embed(text=["dog barking", "cat meowing", "bird singing"])
print(result.text_embeds.shape)  # [3, 512]

# Zero-shot classification
result = ma.embed(
    audio="sound.wav",
    text=["dog barking", "cat meowing", "bird singing"],
    return_similarity=True,
)
print(f"Best match: {result.best_match()}")
print(f"Similarity scores: {result.similarity}")
```

**Available models:** `clap-htsat-fused` (default), `clap-htsat-unfused`

## Voice Activity Detection

Detect speech segments in audio.

```python
import mlx_audio as ma

# Basic speech detection
result = ma.detect_speech("recording.wav")

for segment in result.segments:
    print(f"Speech: {segment.start:.2f}s - {segment.end:.2f}s")

# Custom threshold (higher = more strict)
result = ma.detect_speech("audio.wav", threshold=0.7)

# Get per-frame probabilities
result = ma.detect_speech("audio.wav", return_probabilities=True)
print(result.probabilities.shape)  # Per-frame speech probability

# Adjust segment merging behavior
result = ma.detect_speech(
    "audio.wav",
    min_speech_duration=0.5,   # Ignore segments < 0.5s
    min_silence_duration=0.3,  # Merge segments with < 0.3s gap
)
```

**Available models:** `silero-vad` (default), `silero-vad-8k`

## Audio Enhancement

Remove noise and improve audio quality.

```python
import mlx_audio as ma

# Neural enhancement (best quality)
result = ma.enhance("noisy_speech.wav")
result.save("clean.wav")

# Spectral gating (no model download needed)
result = ma.enhance("audio.wav", method="spectral")

# Compare before and after
result = ma.enhance("noisy.wav", keep_original=True)
original, enhanced = result.before_after

# Save comparison
import soundfile as sf
sf.write("original.wav", original, result.sample_rate)
sf.write("enhanced.wav", enhanced, result.sample_rate)
```

**Available models:** `deepfilternet2` (default), `deepfilternet2-16k`

## Speaker Diarization

Identify who spoke when in audio.

```python
import mlx_audio as ma

# Basic diarization
result = ma.diarize("meeting.wav")

for segment in result.segments:
    print(f"Speaker {segment.speaker}: {segment.start:.1f}s - {segment.end:.1f}s")

# Known number of speakers
result = ma.diarize("interview.wav", num_speakers=2)

# Auto-detect with constraints
result = ma.diarize(
    "podcast.wav",
    min_speakers=2,
    max_speakers=5,
)

# Combined with transcription
transcript = ma.transcribe("meeting.wav")
result = ma.diarize("meeting.wav", transcription=transcript)

for segment in result.segments:
    print(f"{segment.speaker}: {segment.text}")

# Get speaker embeddings for identification
result = ma.diarize("audio.wav", return_embeddings=True)
print(result.embeddings.shape)  # [num_speakers, embedding_dim]
```

**Available models:** `ecapa-tdnn` (default)

## Audio Classification

Single-label classification of audio content.

```python
import mlx_audio as ma

# Zero-shot classification with CLAP
result = ma.classify(
    "sound.wav",
    labels=["dog barking", "cat meowing", "bird singing"],
)

print(f"Predicted: {result.predicted_class}")
print(f"Confidence: {result.confidence:.1%}")

# Get top-k predictions
result = ma.classify(
    "audio.wav",
    labels=["speech", "music", "noise", "silence"],
    top_k=3,
)

for label, score in result.top_k_predictions:
    print(f"{label}: {score:.1%}")

# With trained classifier
result = ma.classify("audio.wav", model="./my_classifier")
```

**Available models:** `clap-htsat-fused` (default for zero-shot), custom trained classifiers

## Audio Tagging

Multi-label classification (tagging) of audio content.

```python
import mlx_audio as ma

# Zero-shot tagging with CLAP
result = ma.tag(
    "music.wav",
    tags=["piano", "guitar", "drums", "vocals", "bass", "strings"],
)

print(f"Active tags: {result.tags}")  # Tags above threshold

# Get top-k tags with scores
for tag, score in result.top_k(5):
    print(f"{tag}: {score:.1%}")

# Custom threshold
result = ma.tag("audio.wav", tags=["speech", "music"], threshold=0.3)

# List all tags above threshold with probabilities
for tag, prob in result.above_threshold():
    print(f"{tag}: {prob:.1%}")

# With trained tagger (e.g., AudioSet)
result = ma.tag("audio.wav", model="./audioset_tagger")
```

**Available models:** `clap-htsat-fused` (default for zero-shot), custom trained taggers

## Working with Audio Arrays

All functions accept multiple audio input formats:

```python
import mlx_audio as ma
import numpy as np
import mlx.core as mx

# From file path
result = ma.transcribe("audio.wav")

# From numpy array (assume 16kHz mono)
audio_np = np.random.randn(16000 * 5)  # 5 seconds
result = ma.transcribe(audio_np)

# From MLX array
audio_mx = mx.array(audio_np)
result = ma.transcribe(audio_mx)

# From dict with sample rate
audio_dict = {"array": audio_np, "sample_rate": 44100}
result = ma.transcribe(audio_dict)
```

## Next Steps

- [API Reference](api/functional.md) — Complete function signatures and parameters
- [Primitives](api/primitives.md) — Low-level DSP operations
- [Models](api/models.md) — All available pre-trained models
- [Training Tutorial](tutorials/training-custom-model.md) — Train your own models
