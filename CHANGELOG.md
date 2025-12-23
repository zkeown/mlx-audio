# Changelog

All notable changes to mlx-audio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-23

Initial stable release of mlx-audio, a complete audio ML toolkit for Apple Silicon.

### Added

#### High-Level API (Python)

- `separate()` - Source separation using HTDemucs (vocals, drums, bass, other)
- `transcribe()` - Speech-to-text using Whisper with word-level timestamps
- `generate()` - Text-to-music generation using MusicGen
- `speak()` - Text-to-speech synthesis using Parler-TTS
- `embed()` - Audio/text embeddings using CLAP for similarity search
- `detect_speech()` - Voice activity detection using Silero VAD
- `enhance()` - Audio denoising using DeepFilterNet
- `diarize()` - Speaker diarization using ECAPA-TDNN
- `classify()` - Zero-shot audio classification via CLAP
- `tag()` - Multi-label audio tagging via CLAP

#### DSP Primitives (Python)

- 40+ audio processing operations optimized for Apple Silicon
- `stft()` / `istft()` - Short-time Fourier transform
- `melspectrogram()` - Mel-scale spectrogram
- `mfcc()` - Mel-frequency cepstral coefficients
- `resample()` - Sample rate conversion
- `griffinlim()` - Phase reconstruction
- Optional C++/Metal acceleration for performance-critical operations

#### Training Framework (Python)

- `TrainModule` - Lightning-like base class for trainable models
- `Trainer` - Training loop with callbacks, logging, and checkpointing
- `OptimizerConfig` - Unified optimizer configuration
- Built-in callbacks: early stopping, model checkpoint, learning rate scheduling
- Logger integrations: WandB, TensorBoard, MLflow

#### Data Loading (Python)

- `DataLoader` - PyTorch-compatible with async prefetching
- `Dataset` - Base dataset class with transforms
- `StreamingDataset` - Memory-efficient streaming for large datasets

#### Models (Python)

- **HTDemucs** - Hybrid transformer for music source separation
- **Whisper** - OpenAI's speech recognition (tiny to large-v3-turbo)
- **MusicGen** - Meta's text-to-music generation (small, medium, large)
- **CLAP** - Contrastive language-audio pretraining
- **EnCodec** - Neural audio codec with RVQ
- **Parler-TTS** - Describable text-to-speech
- **Silero VAD** - Voice activity detection
- **DeepFilterNet** - Speech enhancement
- **ECAPA-TDNN** - Speaker embeddings

#### Swift Package

- **MLXAudioPrimitives** - Core DSP operations (STFT, mel, MFCC)
- **MLXAudioModels** - Pre-built models:
  - HTDemucs (source separation)
  - Whisper (speech recognition)
  - MusicGen (music generation)
  - CLAP (audio embeddings)
  - EnCodec (neural codec)
  - Banquet (query-based separation)
- **MLXAudioStreaming** - Real-time audio with AVFoundation
- **MLXAudioTraining** - On-device fine-tuning support
- **MLXAudio** - High-level API (in progress)

#### Infrastructure

- HuggingFace Hub integration for model weights
- Lazy loading and caching system
- Parity tests against reference implementations (librosa, torchaudio)
- CI/CD with GitHub Actions (Python and Swift)

### Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+ / Swift 6.0+
- MLX 0.30.0+ / mlx-swift 0.10.0+

### Notes

- MusicGen and EnCodec model weights are licensed CC-BY-NC 4.0 (non-commercial only)
- See THIRD_PARTY_LICENSES.md for complete model weight attribution
