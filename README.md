# mlx-audio

[![Python CI](https://github.com/zkeown/mlx-audio/actions/workflows/python-ci.yml/badge.svg)](https://github.com/zkeown/mlx-audio/actions/workflows/python-ci.yml)
[![Swift](https://github.com/zkeown/mlx-audio/actions/workflows/swift.yml/badge.svg)](https://github.com/zkeown/mlx-audio/actions/workflows/swift.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20iOS-lightgrey.svg)](https://www.apple.com/macos/)

**Complete audio ML toolkit for Apple Silicon**

Speech recognition, music generation, source separation, audio embeddings, and more — all running natively on your Mac using [MLX](https://github.com/ml-explore/mlx).

## Installation

### Python

```bash
pip install git+https://github.com/zkeown/mlx-audio.git#subdirectory=python
```

For development:

```bash
git clone https://github.com/zkeown/mlx-audio
cd mlx-audio/python
pip install -e ".[dev]"
```

### Swift

Add to your `Package.swift`:

```swift
.package(url: "https://github.com/zkeown/mlx-audio", from: "1.0.0")
```

Then import the libraries you need:

```swift
import MLXAudio           // High-level API
import MLXAudioModels     // Pre-built models
import MLXAudioPrimitives // DSP operations
import MLXAudioStreaming  // Real-time audio
import MLXAudioTraining   // On-device training
```

## Quick Start

### Transcribe Speech

```python
import mlx_audio

result = mlx_audio.transcribe("speech.wav")
print(result.text)
# Save as subtitles
result.save("subtitles.srt", format="srt")
```

### Separate Music Stems

```python
stems = mlx_audio.separate("song.mp3")
stems.vocals.save("vocals.wav")
stems.drums.save("drums.wav")
stems.bass.save("bass.wav")
stems.other.save("other.wav")
```

### Generate Music

```python
audio = mlx_audio.generate("upbeat jazz piano solo", duration=10.0)
audio.save("jazz.wav")
```

### Text-to-Speech

```python
audio = mlx_audio.speak(
    "Hello, welcome to mlx-audio!",
    description="A warm female voice, speaking clearly"
)
audio.save("greeting.wav")
```

### Audio Embeddings

```python
# Zero-shot audio classification
result = mlx_audio.embed(
    audio="sound.wav",
    text=["dog barking", "cat meowing", "bird singing"],
    return_similarity=True
)
print(result.best_match())  # "dog barking"
```

### Voice Activity Detection

```python
result = mlx_audio.detect_speech("recording.wav")
for segment in result.segments:
    print(f"Speech: {segment.start:.2f}s - {segment.end:.2f}s")
```

### Enhance Audio

```python
result = mlx_audio.enhance("noisy_speech.wav")
result.save("clean_speech.wav")
```

### Speaker Diarization

```python
result = mlx_audio.diarize("meeting.wav")
for segment in result.segments:
    print(f"Speaker {segment.speaker}: {segment.start:.2f}s - {segment.end:.2f}s")
```

### Audio Classification

```python
result = mlx_audio.classify("sound.wav", labels=["music", "speech", "noise"])
print(f"Predicted: {result.predicted_class} ({result.confidence:.1%})")
```

### Audio Tagging

```python
result = mlx_audio.tag("music.wav", tags=["jazz", "piano", "upbeat", "slow"])
print(f"Active tags: {result.tags}")  # ["jazz", "piano", "upbeat"]
```

## Feature Matrix

| Feature | Python | Swift |
|---------|:------:|:-----:|
| Source Separation (HTDemucs) | Yes | Yes |
| Speech Recognition (Whisper) | Yes | Yes |
| Music Generation (MusicGen) | Yes | Yes |
| Audio Embeddings (CLAP) | Yes | Yes |
| Neural Codec (EnCodec) | Yes | Yes |
| Text-to-Speech (Parler-TTS) | Yes | Planned |
| Voice Activity Detection | Yes | Planned |
| Audio Enhancement | Yes | Planned |
| Speaker Diarization | Yes | Planned |
| DSP Primitives | 40+ | 20+ |
| Training Framework | Yes | Yes |
| Real-time Streaming | Yes | Yes |

## API Consistency

Python and Swift APIs follow the same design patterns:

| Python | Swift | Notes |
|--------|-------|-------|
| `mlx_audio.separate()` | `MLXAudio.separate()` | Same parameters |
| `mlx_audio.transcribe()` | `MLXAudio.transcribe()` | Same parameters |
| `mlx_audio.generate()` | `MLXAudio.generate()` | Same parameters |
| `snake_case` | `camelCase` | Naming convention |

## Architecture

```
mlx-audio/
├── python/
│   └── mlx_audio/
│       ├── primitives/    # Audio DSP (STFT, Mel, MFCC, etc.)
│       ├── data/          # DataLoader and datasets
│       ├── train/         # Training framework
│       ├── models/        # Model implementations
│       ├── functional/    # High-level API
│       ├── hub/           # Model registry and caching
│       └── streaming/     # Real-time audio I/O
│
└── swift/
    └── Sources/
        ├── MLXAudio/           # High-level API
        ├── MLXAudioPrimitives/ # DSP operations
        ├── MLXAudioModels/     # Pre-built models
        ├── MLXAudioStreaming/  # Real-time audio
        └── MLXAudioTraining/   # On-device training
```

## Requirements

- **Hardware**: macOS with Apple Silicon (M1/M2/M3/M4) or iOS device with A14+
- **Python**: 3.11+ with MLX 0.30.0+
- **Swift**: 6.0+ with mlx-swift 0.10.0+

## Model Weights License Notice

This library uses pre-trained model weights from various sources. Most models use permissive licenses (MIT, Apache 2.0), but some have restrictions:

- **MusicGen** and **EnCodec** (Meta): CC-BY-NC 4.0 — **non-commercial use only**
- **Whisper** (OpenAI): MIT
- **HTDemucs** (Meta): MIT
- **CLAP** (LAION): Apache 2.0
- **Parler-TTS**: Apache 2.0

See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for complete attribution.

## Documentation

- [Python Documentation](python/docs/index.md)
- [API Reference](python/docs/api/functional.md)
- [Training Tutorial](python/docs/tutorials/training-custom-model.md)
- [Contributing Guide](python/docs/contributing.md)

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.
