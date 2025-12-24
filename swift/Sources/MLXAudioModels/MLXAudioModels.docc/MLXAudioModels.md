# ``MLXAudioModels``

Pre-trained audio models for MLX.

## Overview

MLXAudioModels provides implementations of state-of-the-art audio models, all optimized for Apple Silicon using MLX.

### Available Models

| Model | Task | Description |
|-------|------|-------------|
| HTDemucs | Source Separation | Split audio into vocals, drums, bass, other |
| Whisper | Transcription | Speech-to-text with timestamps |
| CLAP | Embeddings | Audio-text embeddings and similarity |
| MusicGen | Generation | Text-to-music generation |
| Parler-TTS | Text-to-Speech | Natural speech synthesis |
| SileroVAD | VAD | Voice activity detection |
| DeepFilterNet | Enhancement | Audio noise reduction |
| EnCodec | Codec | Neural audio compression |

### Loading Models

Models are loaded from HuggingFace Hub and cached locally:

```swift
import MLXAudioModels

// Load HTDemucs
let demucs = try await HTDemucs.fromPretrained("htdemucs_ft")

// Load Whisper
let whisper = try await Whisper.fromPretrained("whisper-large-v3-turbo")

// Load CLAP
let clap = try await CLAP.fromPretrained("clap-htsat-fused")
```

### Model Configuration

Each model has a configuration type:

```swift
// Whisper configuration
let config = WhisperConfig(
    sampleRate: 16000,
    nMels: 128,
    nFFT: 400
)

// HTDemucs configuration
let config = HTDemucsConfig(
    sources: ["drums", "bass", "other", "vocals"],
    sampleRate: 44100
)
```

## Topics

### Source Separation

- ``HTDemucs``
- ``HTDemucsConfig``
- ``Banquet``
- ``BanquetConfig``

### Speech Recognition

- ``Whisper``
- ``WhisperConfig``
- ``WhisperTokenizer``

### Audio-Text

- ``CLAP``
- ``CLAPConfig``
- ``CLAPAudioConfig``
- ``CLAPTextConfig``

### Audio Generation

- ``MusicGen``
- ``MusicGenConfig``

### Text-to-Speech

- ``ParlerTTS``
- ``ParlerTTSConfig``

### Voice Activity Detection

- ``SileroVAD``
- ``SileroVADConfig``

### Audio Enhancement

- ``DeepFilterNet``
- ``DeepFilterNetConfig``

### Audio Codec

- ``EnCodec``
- ``EnCodecConfig``

### Model Loading

- ``AudioModel``
- ``ModelCache``
- ``PretrainedModel``
