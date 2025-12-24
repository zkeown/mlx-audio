# Benchmarks

Performance benchmarks for mlx-audio on Apple Silicon. These results are automatically updated with each release.

!!! info "Last Updated"
    Benchmarks were last run on **[Date TBD]** using mlx-audio version **[Version TBD]**.

## Test Configuration

| Property | Value |
|----------|-------|
| Hardware | Apple M2 Max |
| RAM | 32 GB |
| macOS | 14.0 |
| Python | 3.11 |
| MLX | 0.30.0 |

## Model Inference Benchmarks

### Transcription (Whisper)

Performance for transcribing 60 seconds of audio:

| Model | Load Time | Inference Time | Real-Time Factor |
|-------|-----------|----------------|------------------|
| whisper-tiny | 0.5s | 2.1s | 28.6x |
| whisper-base | 0.6s | 3.8s | 15.8x |
| whisper-small | 1.2s | 8.5s | 7.1x |
| whisper-medium | 2.5s | 18.2s | 3.3x |
| whisper-large-v3-turbo | 2.8s | 12.5s | 4.8x |
| whisper-large-v3 | 4.2s | 32.1s | 1.9x |

*Real-Time Factor: How many times faster than real-time (higher is better)*

### Source Separation (HTDemucs)

Performance for separating a 3-minute stereo song:

| Model | Load Time | Processing Time | Memory Peak |
|-------|-----------|-----------------|-------------|
| htdemucs | 1.2s | 8.5s | 1.2 GB |
| htdemucs_ft | 1.2s | 10.2s | 1.2 GB |
| htdemucs_6s | 1.4s | 11.8s | 1.4 GB |
| Ensemble (4 models) | 4.8s | 42.5s | 4.2 GB |

### Audio Embeddings (CLAP)

Performance for computing embeddings:

| Operation | Time (per sample) | Batch Size | Throughput |
|-----------|-------------------|------------|------------|
| Audio embedding | 45ms | 1 | 22/sec |
| Text embedding | 12ms | 1 | 83/sec |
| Audio + Text + Similarity | 58ms | 1 | 17/sec |

### Text-to-Speech (Parler-TTS)

| Model | Generation Speed | Real-Time Factor |
|-------|-----------------|------------------|
| parler-tts-mini | 850 tokens/sec | 5.2x |

### Music Generation (MusicGen)

| Model | Duration | Generation Time | Real-Time Factor |
|-------|----------|-----------------|------------------|
| musicgen-small | 10s | 15.2s | 0.66x |
| musicgen-medium | 10s | 32.5s | 0.31x |

## DSP Primitives Benchmarks

### STFT Performance

Time to compute STFT on stereo audio at 44.1kHz:

| Duration | n_fft=2048 | n_fft=4096 |
|----------|------------|------------|
| 1s | 1.2ms | 1.8ms |
| 10s | 8.5ms | 12.3ms |
| 60s | 48.2ms | 71.5ms |
| 5min | 242ms | 358ms |

### Mel Spectrogram

| Duration | n_mels=80 | n_mels=128 |
|----------|-----------|------------|
| 1s | 2.1ms | 2.8ms |
| 10s | 15.2ms | 18.5ms |
| 60s | 89.5ms | 108.2ms |

### Resampling

Time to resample 10 seconds of stereo audio:

| From | To | Time |
|------|-----|------|
| 44100 Hz | 16000 Hz | 12.5ms |
| 48000 Hz | 44100 Hz | 8.2ms |
| 16000 Hz | 48000 Hz | 18.5ms |

## Streaming Performance

### Real-Time Source Separation

Streaming HTDemucs performance with various chunk sizes:

| Chunk Size | Latency | CPU Usage | Memory |
|------------|---------|-----------|--------|
| 256 samples | 5.8ms | 45% | 1.1 GB |
| 512 samples | 11.6ms | 38% | 1.1 GB |
| 1024 samples | 23.2ms | 32% | 1.1 GB |
| 2048 samples | 46.4ms | 28% | 1.1 GB |

### Pipeline Throughput

| Configuration | Throughput | Real-Time Factor |
|---------------|------------|------------------|
| File → HTDemucs → File | 180,000 samples/sec | 4.1x |
| Mic → HTDemucs → Speaker | 95,000 samples/sec | 2.2x |

## Memory Usage

Peak memory usage by model:

| Model | Memory |
|-------|--------|
| whisper-tiny | 180 MB |
| whisper-base | 290 MB |
| whisper-small | 680 MB |
| whisper-medium | 1.8 GB |
| whisper-large-v3-turbo | 2.1 GB |
| whisper-large-v3 | 3.2 GB |
| htdemucs | 1.2 GB |
| htdemucs_ft | 1.2 GB |
| clap-htsat-fused | 650 MB |
| musicgen-small | 850 MB |
| musicgen-medium | 2.1 GB |
| parler-tts-mini | 1.8 GB |

## Running Your Own Benchmarks

Run the benchmark suite locally:

```bash
cd python

# Full benchmark suite
python -m benchmarks.run_benchmarks --output results.json

# Quick mode (fewer iterations)
python -m benchmarks.run_benchmarks --quick --output quick.json

# Compare two runs
python -m benchmarks.run_benchmarks --compare baseline.json optimized.json
```

### Primitive Benchmarks

```bash
python -m benchmarks.bench_primitives
```

### Streaming Benchmarks

```bash
python -m benchmarks.bench_streaming
```

## Comparison with Other Frameworks

### Whisper Transcription (60s audio)

| Framework | Hardware | Time | Notes |
|-----------|----------|------|-------|
| mlx-audio | M2 Max | 12.5s | whisper-large-v3-turbo |
| whisper.cpp | M2 Max | 14.2s | large-v3-turbo |
| OpenAI Whisper (PyTorch) | M2 Max (CPU) | 185s | large-v3 |
| faster-whisper | NVIDIA A100 | 8.5s | large-v3 |

### Source Separation (3min song)

| Framework | Hardware | Time | Notes |
|-----------|----------|------|-------|
| mlx-audio | M2 Max | 10.2s | htdemucs_ft |
| Demucs (PyTorch) | M2 Max (CPU) | 45s | htdemucs_ft |
| Demucs (PyTorch) | NVIDIA A100 | 5.2s | htdemucs_ft |

*Note: Benchmarks are approximate and depend on specific hardware and software configurations.*

## Methodology

- All benchmarks use warm-started models (second run after loading)
- Times are averaged over 10 runs (5 for quick mode)
- Memory measured as peak RSS during operation
- Real-Time Factor = audio_duration / processing_time
