# ``MLXAudioStreaming``

Real-time audio streaming for MLX.

## Overview

MLXAudioStreaming provides a framework for real-time audio processing, including buffering, chunked processing, and integration with AVFoundation for audio I/O.

### Quick Start

```swift
import MLXAudioStreaming
import MLXAudioModels

// Create a streaming pipeline for real-time separation
let model = try await HTDemucs.fromPretrained("htdemucs_ft")

let pipeline = StreamingPipeline(
    source: MicrophoneSource(),
    processor: HTDemucsStreamProcessor(model: model),
    sink: SpeakerSink(stemIndex: 3)  // Play vocals only
)

// Start processing
try await pipeline.start()

// Stop when done
pipeline.stop()
```

### Architecture

The streaming system is built around three components:

1. **Sources** - Provide audio input (microphone, file, callback)
2. **Processors** - Transform audio (separation, effects, etc.)
3. **Sinks** - Handle output (speaker, file, callback)

These connect via a **Pipeline** that manages data flow.

## Topics

### Pipeline

- ``StreamingPipeline``
- ``StreamState``
- ``StreamStats``

### Sources

- ``AudioSource``
- ``MicrophoneSource``
- ``FileSource``
- ``CallbackSource``

### Sinks

- ``AudioSink``
- ``SpeakerSink``
- ``FileSink``
- ``CallbackSink``

### Processors

- ``StreamProcessor``
- ``HTDemucsStreamProcessor``
- ``GainProcessor``

### Buffering

- ``AudioRingBuffer``
- ``StreamChunk``

### Context

- ``StreamingContext``
