# Getting Started with MLXAudio

Set up MLXAudio and run your first audio processing task.

## Overview

This guide walks you through installing MLXAudio and using it to transcribe speech, separate music, and more.

## Installation

Add MLXAudio to your Swift Package Manager dependencies:

```swift
dependencies: [
    .package(url: "https://github.com/zkeown/mlx-audio.git", from: "1.0.0")
]
```

Then add it to your target:

```swift
targets: [
    .target(
        name: "YourApp",
        dependencies: ["MLXAudio"]
    )
]
```

## Requirements

- macOS 14.0+ or iOS 17.0+
- Apple Silicon (M1 or later)
- Xcode 15.0+

## Your First Transcription

Transcribe an audio file to text:

```swift
import MLXAudio

// Transcribe audio file
let result = try await MLXAudio.transcribe("recording.wav")
print(result.text)

// Access segments with timestamps
for segment in result.segments {
    print("[\(segment.start) - \(segment.end)] \(segment.text)")
}
```

## Source Separation

Separate a song into individual stems:

```swift
import MLXAudio

// Separate into vocals, drums, bass, other
let stems = try await MLXAudio.separate("song.mp3")

// Save individual stems
try await stems.vocals.save(to: "vocals.wav")
try await stems.drums.save(to: "drums.wav")
try await stems.bass.save(to: "bass.wav")
try await stems.other.save(to: "other.wav")
```

## Audio Classification

Classify audio using zero-shot learning:

```swift
import MLXAudio

let result = try await MLXAudio.classify(
    "sound.wav",
    labels: ["dog barking", "car horn", "music"]
)

print("Detected: \(result.predictedClass)")
print("Confidence: \(result.confidence)")
```

## Working with AudioData

`AudioData` is the core type for handling audio:

```swift
import MLXAudio

// Load from file
let audio = try AudioData(contentsOf: URL(fileURLWithPath: "audio.wav"))

// Access properties
print("Sample rate: \(audio.sampleRate)")
print("Channels: \(audio.channels)")
print("Duration: \(audio.duration) seconds")

// Save to file
try await audio.save(to: "output.wav")
```

## Model Selection

Each function accepts a model parameter:

```swift
// Use different Whisper models
let fast = try await MLXAudio.transcribe("audio.wav", model: "whisper-tiny")
let accurate = try await MLXAudio.transcribe("audio.wav", model: "whisper-large-v3-turbo")

// Different separation models
let basic = try await MLXAudio.separate("song.mp3", model: "htdemucs")
let finetuned = try await MLXAudio.separate("song.mp3", model: "htdemucs_ft")
```

## Next Steps

- Learn about <doc:SourceSeparation> in depth
- Explore <doc:Transcription> options
- See all available models in ``MLXAudioModels``
