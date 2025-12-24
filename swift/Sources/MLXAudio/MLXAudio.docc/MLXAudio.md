# ``MLXAudio``

Audio machine learning toolkit for Apple Silicon.

## Overview

MLXAudio provides a comprehensive suite of audio processing tools optimized for Apple Silicon using the MLX framework. From speech recognition to music generation, audio enhancement to speaker diarization—all running natively on your Mac or iOS device.

### Quick Start

```swift
import MLXAudio

// Transcribe speech
let result = try await MLXAudio.transcribe("speech.wav")
print(result.text)

// Separate music
let stems = try await MLXAudio.separate("song.mp3")
try await stems.vocals.save(to: "vocals.wav")

// Generate music
let music = try await MLXAudio.generate("upbeat jazz piano")
try await music.save(to: "jazz.wav")
```

### Features

- **10 Pre-trained Models** — Whisper, MusicGen, CLAP, HTDemucs, Parler-TTS, and more
- **High Performance** — Optimized for Apple Silicon with MLX
- **Swift Native** — Modern async/await API with strong typing
- **Cross-Platform** — Works on macOS and iOS

## Topics

### Getting Started

- <doc:GettingStarted>
- <doc:SourceSeparation>
- <doc:Transcription>

### High-Level API

- ``separate(audio:model:stems:)``
- ``transcribe(audio:model:language:)``
- ``generate(prompt:model:duration:)``
- ``embed(audio:text:model:)``
- ``classify(audio:labels:model:)``
- ``tag(audio:tags:model:)``
- ``speak(text:model:description:)``
- ``detectSpeech(audio:model:threshold:)``
- ``enhance(audio:model:)``
- ``diarize(audio:model:)``

### Core Types

- ``AudioData``
- ``SeparationResult``
- ``TranscriptionResult``
- ``GenerationResult``
- ``EmbeddingResult``
- ``ClassificationResult``
- ``TaggingResult``
- ``SpeechResult``
- ``VADResult``
- ``EnhancementResult``
- ``DiarizationResult``
