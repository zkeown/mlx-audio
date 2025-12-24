# ``MLXAudioPrimitives``

Low-level audio DSP operations for MLX.

## Overview

MLXAudioPrimitives provides fundamental audio signal processing operations optimized for Apple Silicon. These primitives are the building blocks for higher-level audio processing tasks.

### Quick Start

```swift
import MLXAudioPrimitives
import MLX

// Compute STFT
let audio = MLXArray(...)  // [channels, samples]
let stft = MLXAudioPrimitives.stft(
    audio,
    nFFT: 2048,
    hopLength: 512,
    windowLength: 2048
)

// Compute mel spectrogram
let mel = MLXAudioPrimitives.melSpectrogram(
    audio,
    sampleRate: 44100,
    nMels: 128
)

// Compute MFCC
let mfcc = MLXAudioPrimitives.mfcc(
    audio,
    sampleRate: 16000,
    nMFCC: 13
)
```

## Topics

### STFT Operations

- ``stft(_:nFFT:hopLength:windowLength:window:center:padMode:)``
- ``istft(_:nFFT:hopLength:windowLength:window:center:length:)``
- ``spectrogram(_:nFFT:hopLength:power:)``
- ``phaseVocoder(_:rate:hopLength:)``

### Mel-Scale Operations

- ``melSpectrogram(_:sampleRate:nFFT:hopLength:nMels:fMin:fMax:)``
- ``melFilterbank(sampleRate:nFFT:nMels:fMin:fMax:)``
- ``hzToMel(_:)``
- ``melToHz(_:)``

### MFCC

- ``mfcc(_:sampleRate:nMFCC:nMels:nFFT:hopLength:)``
- ``delta(_:width:)``

### Window Functions

- ``getWindow(_:length:)``
- ``WindowType``

### Resampling

- ``resample(_:origSR:targetSR:)``

### Utilities

- ``powerToDb(_:ref:amin:topDb:)``
- ``dbToPower(_:ref:)``
- ``amplitudeToDb(_:ref:amin:topDb:)``
- ``dbToAmplitude(_:ref:)``

### Complex Arrays

- ``ComplexArray``
