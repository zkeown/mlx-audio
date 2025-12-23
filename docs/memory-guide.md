# Memory Optimization Guide for MLX-Audio Swift

This guide covers memory optimization strategies for running MLX-Audio models on Apple Silicon devices, from iPhones with 6GB RAM to Mac Studios with 96GB+.

## Table of Contents

1. [Memory Architecture Overview](#memory-architecture-overview)
2. [Per-Model Memory Requirements](#per-model-memory-requirements)
3. [Device-Specific Recommendations](#device-specific-recommendations)
4. [Quantization Guide](#quantization-guide)
5. [Chunked Processing Guide](#chunked-processing-guide)
6. [iOS Integration Guide](#ios-integration-guide)
7. [Troubleshooting Memory Issues](#troubleshooting-memory-issues)

---

## Memory Architecture Overview

### MLX Unified Memory

MLX uses Apple's unified memory architecture where GPU and CPU share the same memory pool. This means:

- No explicit GPU↔CPU transfers needed
- GPU memory usage directly impacts system memory
- Memory pressure affects all components equally

### Memory Components

For each model, memory usage breaks down into:

| Component | Description | Typical % |
|-----------|-------------|-----------|
| **Weights** | Model parameters | 60-80% |
| **KV Cache** | Autoregressive decoder cache | 5-20% |
| **Activations** | Intermediate tensors during forward pass | 10-30% |
| **Buffers** | Input/output buffers, temporary arrays | 5-10% |

### Key APIs

```swift
// Check GPU memory usage
let activeMemory = GPU.activeMemory      // Currently allocated
let peakMemory = GPU.peakMemory          // Peak since last reset
let cacheMemory = GPU.cacheMemory        // MLX cache

// Clear GPU cache
GPU.clearCache()

// Reset peak tracking
GPU.resetPeakMemory()

// Force computation (releases lazy tensors)
eval(tensor1, tensor2)
```

---

## Per-Model Memory Requirements

### Full Precision (float32)

| Model | Variant | Parameters | Memory | Peak Memory |
|-------|---------|------------|--------|-------------|
| **Whisper** | tiny | 39M | 150MB | 200MB |
| | small | 244M | 500MB | 700MB |
| | medium | 769M | 1.5GB | 2GB |
| | large-v3-turbo | 809M | 1.6GB | 2.2GB |
| | large-v3 | 1.55B | 3GB | 4GB |
| **HTDemucs** | htdemucs | 42M | 2GB | 4GB |
| | htdemucs_ft | 42M | 2GB | 4GB |
| | htdemucs_6s | 42M | 2.5GB | 5GB |
| **MusicGen** | small | 300M | 1.2GB | 2GB |
| | medium | 1.5B | 3.5GB | 5GB |
| | large | 3.3B | 7GB | 10GB |
| **CLAP** | htsat-tiny | 30M | 150MB | 300MB |
| | htsat-fused | 200M | 800MB | 1.2GB |
| **EnCodec** | 24kHz | 15M | 100MB | 150MB |
| | 48kHz | 15M | 120MB | 180MB |

### With Quantization

| Model | Config | Memory Reduction | Quality Impact |
|-------|--------|------------------|----------------|
| Whisper large-v3 | int4 | 3GB → 900MB (3.3x) | WER +0.5% |
| Whisper large-v3 | int8 | 3GB → 1.6GB (1.9x) | WER +0.1% |
| HTDemucs | int8 | 2GB → 600MB (3.3x) | SI-SDR -0.3dB |
| MusicGen medium | int4 | 3.5GB → 1GB (3.5x) | Subjective: minimal |
| CLAP fused | int8 | 800MB → 250MB (3.2x) | Cosine sim: 0.995 |

---

## Device-Specific Recommendations

### iPhone (6-8GB RAM)

**Constraints:**
- Available memory: ~2GB for models
- Recommended: 1 model at a time
- Must use quantization for larger models

**Recommended Configuration:**
```swift
let profile = DeviceProfile.phone

// Use the cache with phone settings
let cache = ModelCache(
    maxModels: 1,
    memoryBudgetMB: 2048
)

// Recommended models
let whisperModel = "whisper-small"      // 500MB
let htdemucsModel = "htdemucs"          // 2GB with int8
let clapModel = "clap-htsat-tiny"       // 150MB
```

**Model Combinations:**
| Use Case | Models | Total Memory |
|----------|--------|--------------|
| Transcription only | whisper-small | 500MB |
| Full processing | whisper-small + clap-tiny + encodec | 750MB |
| With separation (quantized) | whisper-small-q4 + htdemucs-q8 | 900MB |

### iPad (8-16GB RAM)

**Constraints:**
- Available memory: ~4GB for models
- Recommended: 2 models simultaneously
- Quantization optional but beneficial

**Recommended Configuration:**
```swift
let profile = DeviceProfile.tablet

let cache = ModelCache(
    maxModels: 2,
    memoryBudgetMB: 4096
)

// Recommended models
let whisperModel = "whisper-medium"     // 1.5GB
let htdemucsModel = "htdemucs_ft"       // 2GB
```

### Mac (16-64GB RAM)

**Constraints:**
- Available memory: ~8GB for models
- Recommended: 4 models simultaneously
- Full precision works well

**Recommended Configuration:**
```swift
let profile = DeviceProfile.mac

let cache = ModelCache(
    maxModels: 4,
    memoryBudgetMB: 8192
)

// Full quality models
let whisperModel = "whisper-large-v3-turbo"
let htdemucsModel = "htdemucs_ft"
let musicgenModel = "musicgen-medium"
```

### Mac Pro/Studio (64GB+ RAM)

**No significant constraints** - use full-size models in full precision.

---

## Quantization Guide

### When to Use Quantization

| Situation | Recommendation |
|-----------|----------------|
| iPhone deployment | Always (int4 or int8) |
| iPad deployment | Recommended (int8) |
| Mac deployment | Optional |
| Real-time requirements | int8 (faster dequantization) |
| Maximum quality | Full precision |

### Quantization Configurations

```swift
// 4-bit quantization (maximum compression)
let int4Config = QuantizationConfig(
    bits: 4,
    groupSize: 64,
    mode: .affine
)

// 8-bit quantization (quality preservation)
let int8Config = QuantizationConfig(
    bits: 8,
    groupSize: 64,
    mode: .affine
)

// Audio-sensitive (skip critical layers)
let audioConfig = QuantizationConfig(
    bits: 8,
    groupSize: 64,
    mode: .affine,
    skipLayers: ["embed", "output", "proj_out"]
)
```

### Per-Model Recommendations

| Model | Recommended Config | Rationale |
|-------|-------------------|-----------|
| Whisper | int4 | Text output tolerant |
| HTDemucs | int8 | Audio quality sensitive |
| MusicGen | int4 | Large model, acceptable degradation |
| CLAP | int8 | Embedding precision matters |
| EnCodec | None | Small model, codec quality critical |

### Quality Validation

```swift
// Measure quantization error
let maxError = ModelQuantizer.maxQuantizationError(
    original: originalWeight,
    quantized: quantizedLayer
)

// Check error thresholds
// Whisper: maxError < 0.1 acceptable
// HTDemucs: maxError < 0.05 preferred
// CLAP: maxError < 0.02 for embeddings
```

---

## Chunked Processing Guide

### Why Use Chunked Processing

Long audio exceeds memory limits. Chunked processing:
- Bounds memory usage regardless of input length
- Enables streaming output
- Provides progress feedback

### Configuration

```swift
// HTDemucs default (6 second chunks)
let htdemucsChunking = ChunkConfig(
    chunkDurationSec: 6.0,
    overlapRatio: 0.25,        // 25% overlap
    windowFunction: .triangular,
    sampleRate: 44100
)

// Memory-constrained iPhone
let phoneChunking = ChunkConfig(
    chunkDurationSec: 3.0,     // Smaller chunks
    overlapRatio: 0.25,
    windowFunction: .triangular,
    maxMemoryMB: 2048,
    sampleRate: 44100
)
```

### Usage

```swift
let processor = ChunkedProcessor(config: .htdemucs)

let output = processor.process(longAudio) { chunk in
    return model(chunk)
} progressCallback: { progress in
    print("Progress: \(progress.progress * 100)%")
}
```

### Memory Estimation

```swift
// Estimate memory for processing
let estimatedMemory = processor.estimateMemory(
    audioLength: 44100 * 180,  // 3 minutes
    channels: 2,
    sources: 4                 // HTDemucs outputs
)
// Returns bytes needed for buffers
```

---

## iOS Integration Guide

### Setup

```swift
import MLXAudioModels

class AudioProcessor {
    private let cache = ModelCache.forCurrentDevice()
    private let pressureHandler = MemoryPressureHandler()

    func setup() async {
        // Start memory pressure monitoring
        await cache.startMemoryPressureMonitoring()
    }
}
```

### AppDelegate Integration

```swift
// AppDelegate.swift
func applicationDidReceiveMemoryWarning(_ application: UIApplication) {
    Task {
        await memoryHandler.handleAppMemoryWarning()
    }
}

func applicationDidEnterBackground(_ application: UIApplication) {
    Task {
        await memoryHandler.handleEnteringBackground()
    }
}
```

### Best Practices

1. **Use budget-based caching:**
   ```swift
   let cache = ModelCache(
       maxModels: 1,
       memoryBudgetMB: 2048
   )
   ```

2. **Track memory per model:**
   ```swift
   let model = try await cache.get(
       id: "whisper-small",
       estimatedMemoryMB: 500
   ) { try loadModel() }
   ```

3. **Clear cache before heavy operations:**
   ```swift
   GPU.clearCache()
   eval()  // Force pending operations
   ```

4. **Use chunked processing for long audio:**
   ```swift
   let processor = ChunkedProcessor.forDevice(.phone)
   ```

---

## Troubleshooting Memory Issues

### Common Issues

#### "Insufficient memory" error

**Symptoms:** App crashes or throws MemoryError

**Solutions:**
1. Use a smaller model variant
2. Enable quantization
3. Reduce chunk size
4. Clear other cached models

```swift
// Check available budget
let remaining = await cache.remainingBudgetMB
if remaining < neededMB {
    await cache.evictLRU()
}
```

#### Memory grows during long processing

**Symptoms:** Memory increases linearly with audio length

**Solutions:**
1. Use chunked processing
2. Add periodic `eval()` calls
3. Use `at[].add()` pattern instead of concatenation

```swift
// Bad: Memory grows
for chunk in chunks {
    output = concatenated([output, processedChunk], axis: -1)
}

// Good: Memory bounded
for chunk in chunks {
    output = output.at[..., offset..<end].add(processedChunk)
    eval(output)
}
```

#### Memory not released after model switch

**Symptoms:** Memory stays high after switching models

**Solutions:**
```swift
// Explicitly evict and clear
await cache.evict(oldModelId)
GPU.clearCache()
```

### Debugging Tools

```swift
// Memory profiler
let profiler = MemoryProfiler(verbose: true)

let (result, profile) = try await profiler.profile("model_forward") {
    return model(input)
}

print("GPU delta: \(profile.gpuDeltaMB)MB")
print("Peak GPU: \(profile.peakGpuMB)MB")

// Get summary
let summary = await profiler.getSummary()
summary.printSummary()
```

```swift
// Memory stats
let handler = MemoryPressureHandler()
let stats = await handler.getMemoryStats()
stats.printSummary()
```

### Performance Tips

1. **Warm up models** before timing-critical operations
2. **Pre-allocate buffers** for known sizes
3. **Use float16** for KV caches (already default)
4. **Batch operations** when possible
5. **Profile before optimizing** - use MemoryProfiler

---

## API Reference

### Key Classes

| Class | Purpose |
|-------|---------|
| `ModelCache` | LRU cache with memory budgeting |
| `DeviceProfile` | Device capability detection |
| `MemoryProfiler` | Memory usage measurement |
| `MemoryPressureHandler` | iOS memory warning handling |
| `ChunkedProcessor` | Long audio processing |
| `QuantizationConfig` | Quantization settings |
| `ModelVariantRegistry` | Model metadata and recommendations |

### Key Functions

```swift
// Memory utilities
formatBytes(UInt64) -> String

// GPU state
GPU.activeMemory -> UInt64
GPU.peakMemory -> UInt64
GPU.clearCache()
GPU.resetPeakMemory()

// Force evaluation
eval(tensor1, tensor2, ...)
```
