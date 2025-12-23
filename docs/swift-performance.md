# Swift MLX-Audio Performance Guide

This document describes the performance optimizations implemented in the Swift mlx-audio models and provides guidance for achieving optimal inference performance on Apple Silicon.

## Performance Targets

| Model | Task | Target | Notes |
|-------|------|--------|-------|
| HTDemucs | 3min song separation | <5s | Full 4-source separation |
| Whisper large-v3-turbo | 30s audio transcription | <3s | Greedy decoding |
| CLAP | 10s audio embedding | <100ms | Audio-only embedding |
| MusicGen | 10s generation | <15s | Medium model, text conditioning |
| EnCodec | 5s decode | <50ms | From discrete codes |

## Key Optimizations

### 1. KV Cache Optimization (Whisper, MusicGen)

**Problem:** Naive KV cache concatenation is O(n) per step, resulting in O(n²) total complexity for n tokens.

**Solution:** Pre-allocated ring buffer pattern with O(1) append using `at[].add()`.

```swift
// Before (O(n) per step):
keys = concatenated([cache.0, keys], axis: 1)

// After (O(1) per step):
keys[layer] = keys[layer].at[0..., startPos..<endPos, 0...].add(
    newK - keys[layer][0..., startPos..<endPos, 0...]
)
```

**Impact:** ~40% speedup for Whisper greedy decode, ~30% for MusicGen generation.

**Files:**
- `swift/Sources/MLXAudioModels/Whisper/Decoding.swift` - `DecodingKVCache`, `greedyDecodeOptimized()`
- `swift/Sources/MLXAudioModels/Whisper/KVCache.swift` - `WhisperKVCache`
- `swift/Sources/MLXAudioModels/MusicGen/MusicGenAttention.swift` - `MusicGenKVCache`

### 2. Cross-Attention Caching (Whisper, MusicGen)

**Problem:** Cross-attention K/V are recomputed each decode step, even though encoder output is fixed.

**Solution:** Compute cross-attention K/V once on first decode step, cache and reuse.

```swift
// First decode step: compute and cache
let (crossOut, crossK, crossV) = crossAttn.forwardOptimized(x, xa: encoderOutput)
crossAttnCache = (crossK, crossV)

// Subsequent steps: reuse cached K/V
let (crossOut, _, _) = crossAttn.forwardOptimized(x, precomputedKV: crossAttnCache)
```

**Impact:** ~20% additional speedup for decode-heavy models.

**Files:**
- `swift/Sources/MLXAudioModels/Whisper/Attention.swift` - `forwardOptimized()`
- `swift/Sources/MLXAudioModels/Whisper/TextDecoder.swift` - `forwardOptimized()`
- `swift/Sources/MLXAudioModels/MusicGen/MusicGenDecoder.swift` - `forwardOptimized()`

### 3. Strategic eval() Placement

**Problem:** Calling `eval()` every decode step adds overhead; not calling it causes memory growth.

**Solution:** Evaluate every N steps (e.g., 8) to balance memory and speed.

```swift
let evalInterval = 8
for step in 0..<maxTokens {
    // ... decode step ...
    if step % evalInterval == 0 {
        eval(logits)
    }
}
```

**Impact:** Reduces eval overhead while preventing unbounded memory growth.

### 4. Format Conversion (HTDemucs)

**Problem:** Transformer layers prefer NHWC format but audio uses NCHW.

**Solution:** Convert at transformer boundary, minimize transposes inside model.

**Files:**
- `swift/Sources/MLXAudioModels/HTDemucs/HTDemucs.swift`

## Benchmarking

### Running Benchmarks

```bash
cd swift

# Run all benchmarks
./run-benchmarks.sh --output results.json

# Quick mode (fewer iterations)
./run-benchmarks.sh --quick

# Single model
./run-benchmarks.sh --model whisper

# Check performance targets
./run-benchmarks.sh --targets

# Compare with baseline
./run-benchmarks.sh --compare baseline.json
```

### CLI Options

| Option | Description |
|--------|-------------|
| `-o, --output <path>` | Save results to JSON file |
| `-c, --compare <path>` | Compare with baseline JSON |
| `-q, --quick` | Quick mode (fewer iterations) |
| `-m, --model <name>` | Run only specific model |
| `-w, --warmup <n>` | Number of warmup iterations |
| `-i, --iterations <n>` | Number of measurement iterations |
| `-t, --targets` | Check performance targets |

## Memory Optimization

### Using ChunkedProcessor

For long audio that exceeds memory, use chunked processing:

```swift
let processor = ChunkedProcessor.forHTDemucs()
let output = processor.process(audio) { chunk in
    return model(chunk)
} progressCallback: { progress in
    print("Progress: \(progress.progress * 100)%")
}
```

**Configurations:**
- `.htdemucs` - 6s chunks, 25% overlap
- `.whisper` - 30s chunks, 10% overlap
- `.phoneConstrained` - 3s chunks for memory-limited devices

### Memory Profiling

```swift
let profiler = MemoryProfiler()
profiler.start()

// ... run inference ...

let snapshot = MemoryProfiler.capture()
print("GPU Active: \(formatBytes(snapshot.gpuActive))")
print("GPU Peak: \(formatBytes(snapshot.gpuPeak))")
```

## Model-Specific Notes

### Whisper

- Use `greedyDecodeOptimized()` for fastest inference (enabled by default in `transcribe()`)
- For beam search, the standard path is still used (optimization TODO)
- Language detection adds one extra forward pass

### MusicGen

- Pre-allocate `MusicGenKVCache` with expected max length for best performance
- Cross-attention cache is populated on first decode step automatically
- Use `forwardOptimized()` path on decoder for O(1) cache updates

### HTDemucs

- Chunked processing is essential for songs >30s on memory-constrained devices
- Overlap-add blending uses triangular window for smooth transitions
- NCHW→NHWC conversion happens at transformer boundaries

### CLAP

- Audio-only embedding is fastest path
- Text+audio joint embedding requires two encoder passes
- Consider batching multiple audio clips for throughput

### EnCodec

- RVQ decode is highly parallelizable
- Encode is heavier due to convolution stack
- Consider streaming decode for real-time applications

## Hardware Recommendations

| Device | Recommendation |
|--------|----------------|
| iPhone | Use `.phoneConstrained` chunk config, quantized models |
| iPad | Standard chunk sizes, FP16 precision |
| M1 Mac | Full size chunks, all models supported |
| M2/M3 Pro/Max | Larger batch sizes, longer sequences |

## Profiling with Instruments

1. Build in Release mode: `swift build -c release`
2. Open Instruments.app
3. Select "Metal System Trace" template
4. Run your benchmark executable
5. Look for:
   - GPU utilization (should be >80%)
   - Memory pressure events
   - Shader compilation stalls (first run only)

## Future Optimizations

- [ ] Beam search with pre-allocated cache
- [ ] Fused attention kernels for specific head counts
- [ ] Dynamic batching for throughput optimization
- [ ] Quantized model support (INT4/INT8)
- [ ] Streaming decode for real-time applications
