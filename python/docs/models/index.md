# Models Overview

mlx-audio provides pre-trained models for a wide range of audio tasks, all optimized for Apple Silicon. Each model is accessible through a simple high-level API function.

## Available Models

| Task | Function | Default Model | Description |
|------|----------|---------------|-------------|
| Source Separation | `separate()` | htdemucs_ft | Split audio into stems (vocals, drums, bass, other) |
| Transcription | `transcribe()` | whisper-large-v3-turbo | Speech-to-text with timestamps |
| Audio Embeddings | `embed()` | clap-htsat-fused | Audio-text embeddings for similarity/search |
| Classification | `classify()` | clap-htsat-fused | Zero-shot audio classification |
| Tagging | `tag()` | clap-htsat-fused | Multi-label audio tagging |
| Music Generation | `generate()` | musicgen-medium | Text-to-music generation |
| Text-to-Speech | `speak()` | parler-tts-mini | Generate speech from text |
| Voice Activity Detection | `detect_speech()` | silero-vad | Detect speech segments |
| Audio Enhancement | `enhance()` | deepfilternet2 | Noise reduction and enhancement |
| Speaker Diarization | `diarize()` | ecapa-tdnn | Identify who spoke when |

## Quick Start

All models use the same simple pattern:

```python
import mlx_audio as ma

# Each task has a dedicated function
result = ma.transcribe("audio.wav")    # Speech-to-text
result = ma.separate("song.mp3")       # Source separation
result = ma.generate("jazz piano")     # Music generation
```

## Model Loading

Models are automatically downloaded from HuggingFace Hub on first use and cached locally. You can specify alternative model variants:

```python
# Use a different model variant
result = ma.transcribe("audio.wav", model="whisper-tiny")
result = ma.separate("song.mp3", model="htdemucs")
```

## Memory Considerations

Different models have different memory requirements:

| Model | Approximate Memory |
|-------|-------------------|
| whisper-tiny | ~40 MB |
| whisper-base | ~75 MB |
| whisper-small | ~250 MB |
| whisper-medium | ~750 MB |
| whisper-large-v3-turbo | ~1.5 GB |
| htdemucs | ~80 MB |
| htdemucs_ft | ~80 MB |
| musicgen-small | ~300 MB |
| musicgen-medium | ~1.5 GB |
| parler-tts-mini | ~900 MB |

See the [Memory Optimization](../advanced/memory.md) guide for tips on managing memory usage.

## Model Guides

Each model has a dedicated guide with detailed usage examples:

- [HTDemucs](htdemucs.md) - Source separation
- [Whisper](whisper.md) - Transcription
- [CLAP](clap.md) - Audio-text embeddings
- [MusicGen](musicgen.md) - Music generation
- [Parler-TTS](parler-tts.md) - Text-to-speech
- [SileroVAD](silero-vad.md) - Voice activity detection
- [DeepFilterNet](deepfilternet.md) - Audio enhancement
- [EnCodec](encodec.md) - Neural audio codec
