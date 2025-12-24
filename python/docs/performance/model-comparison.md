# Model Comparison

This guide helps you choose the right model variant for your use case, balancing speed, quality, and memory usage.

## Transcription Models (Whisper)

### Overview

| Model | Parameters | Speed | Quality | Memory | Best For |
|-------|------------|-------|---------|--------|----------|
| whisper-tiny | 39M | Fastest | Good | 180 MB | Real-time, previews |
| whisper-base | 74M | Very Fast | Good | 290 MB | Quick transcription |
| whisper-small | 244M | Fast | Better | 680 MB | Production use |
| whisper-medium | 769M | Moderate | Very Good | 1.8 GB | High accuracy |
| whisper-large-v3-turbo | 809M | Fast | Excellent | 2.1 GB | **Recommended** |
| whisper-large-v3 | 1.5B | Slow | Best | 3.2 GB | Maximum accuracy |

### Decision Guide

```
Need real-time processing?
├── Yes → whisper-tiny or whisper-base
└── No → Need best quality?
    ├── Yes → whisper-large-v3-turbo (recommended)
    └── No → whisper-small (good balance)
```

### Quality Benchmarks

Word Error Rate (WER) on LibriSpeech test-clean:

| Model | WER |
|-------|-----|
| whisper-tiny | 7.8% |
| whisper-base | 5.4% |
| whisper-small | 3.4% |
| whisper-medium | 2.9% |
| whisper-large-v3-turbo | 2.1% |
| whisper-large-v3 | 2.0% |

### Usage Examples

```python
import mlx_audio as ma

# Quick preview
result = ma.transcribe("audio.wav", model="whisper-tiny")

# Production (recommended)
result = ma.transcribe("audio.wav", model="whisper-large-v3-turbo")

# Maximum accuracy
result = ma.transcribe("audio.wav", model="whisper-large-v3", beam_size=5)
```

## Source Separation Models (HTDemucs)

### Overview

| Model | Stems | Speed | Quality | Memory | Best For |
|-------|-------|-------|---------|--------|----------|
| htdemucs | 4 | Fastest | Good | 1.2 GB | Quick separation |
| htdemucs_ft | 4 | Fast | Better | 1.2 GB | **Recommended** |
| htdemucs_6s | 6 | Moderate | Better | 1.4 GB | Guitar/piano extraction |
| Ensemble | 4 | Slow | Best | 4.2 GB | Critical applications |

### Stem Types

**4-stem models:**
- vocals, drums, bass, other

**6-stem model (htdemucs_6s):**
- vocals, drums, bass, guitar, piano, other

### Quality Benchmarks

Signal-to-Distortion Ratio (SDR) on MUSDB18-HQ test set:

| Model | Vocals | Drums | Bass | Other | Average |
|-------|--------|-------|------|-------|---------|
| htdemucs | 7.8 | 8.2 | 7.1 | 5.8 | 7.2 |
| htdemucs_ft | 8.5 | 8.9 | 7.8 | 6.2 | 7.9 |
| Ensemble | 9.2 | 9.5 | 8.4 | 6.8 | 8.5 |

*Higher SDR is better (measured in dB)*

### Decision Guide

```
Need guitar/piano stems?
├── Yes → htdemucs_6s
└── No → Quality is critical?
    ├── Yes → htdemucs_ft with ensemble=True
    └── No → htdemucs_ft (recommended)
```

### Usage Examples

```python
import mlx_audio as ma

# Standard separation (recommended)
result = ma.separate("song.mp3", model="htdemucs_ft")

# With guitar and piano
result = ma.separate("song.mp3", model="htdemucs_6s")

# Best quality (4x slower)
result = ma.separate("song.mp3", ensemble=True)
```

## Audio Embedding Models (CLAP)

### Overview

| Model | Embedding Dim | Speed | Memory | Best For |
|-------|---------------|-------|--------|----------|
| clap-htsat-fused | 512 | Fast | 650 MB | General use |

### Capabilities

- Zero-shot audio classification
- Audio-text similarity search
- Multi-label audio tagging
- Audio embedding for downstream tasks

### Usage Examples

```python
import mlx_audio as ma

# Classification
result = ma.classify("sound.wav", labels=["dog", "cat", "bird"])

# Embeddings
result = ma.embed(audio="audio.wav")

# Tagging
result = ma.tag("music.wav", tags=["jazz", "piano", "vocals"])
```

## Text-to-Speech Models (Parler-TTS)

### Overview

| Model | Speed | Quality | Memory | Best For |
|-------|-------|---------|--------|----------|
| parler-tts-mini | Fast | Good | 1.8 GB | General TTS |

### Features

- Natural-sounding speech
- Voice description control
- Speed adjustment

### Usage Examples

```python
import mlx_audio as ma

# Basic TTS
result = ma.speak("Hello, world!")

# With voice description
result = ma.speak(
    "Welcome to the show.",
    description="A warm, friendly female voice"
)
```

## Music Generation Models (MusicGen)

### Overview

| Model | Parameters | Speed | Quality | Memory | Best For |
|-------|------------|-------|---------|--------|----------|
| musicgen-small | 300M | Fast | Good | 850 MB | Quick generation |
| musicgen-medium | 1.5B | Moderate | Better | 2.1 GB | **Recommended** |

### Decision Guide

```
Need quick previews?
├── Yes → musicgen-small
└── No → musicgen-medium (better quality)
```

### Usage Examples

```python
import mlx_audio as ma

# Quick generation
result = ma.generate("jazz piano", model="musicgen-small", duration=5.0)

# Better quality
result = ma.generate("jazz piano", model="musicgen-medium", duration=10.0)
```

## Voice Activity Detection (SileroVAD)

### Overview

| Model | Speed | Accuracy | Memory | Best For |
|-------|-------|----------|--------|----------|
| silero-vad | Very Fast | Excellent | 50 MB | All VAD tasks |

### Usage Examples

```python
import mlx_audio as ma

# Detect speech segments
result = ma.detect_speech("audio.wav")

# Adjust sensitivity
result = ma.detect_speech("audio.wav", threshold=0.3)  # More sensitive
```

## Audio Enhancement (DeepFilterNet)

### Overview

| Model | Speed | Quality | Memory | Best For |
|-------|-------|---------|--------|----------|
| deepfilternet2 | Fast | Excellent | 200 MB | Noise reduction |

### Usage Examples

```python
import mlx_audio as ma

# Enhance noisy audio
result = ma.enhance("noisy.wav")
result.save("clean.wav")
```

## Summary: Recommended Models

For most use cases, use these defaults:

| Task | Recommended Model | Why |
|------|-------------------|-----|
| Transcription | whisper-large-v3-turbo | Best speed/quality ratio |
| Separation | htdemucs_ft | Fine-tuned, good quality |
| Classification | clap-htsat-fused | Only option, excellent |
| TTS | parler-tts-mini | Only option, good quality |
| Music Generation | musicgen-medium | Better quality |
| VAD | silero-vad | Only option, excellent |
| Enhancement | deepfilternet2 | Only option, excellent |

## Hardware Requirements

### Minimum (8GB RAM)

Can run: whisper-tiny/base/small, htdemucs, silero-vad, clap

### Recommended (16GB RAM)

Can run: All models except whisper-large-v3, ensemble separation

### Optimal (32GB+ RAM)

Can run: All models including ensemble separation
