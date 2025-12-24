# Memory Optimization

This guide covers techniques for managing memory when working with mlx-audio, especially for large audio files or memory-constrained systems.

## Understanding Memory Usage

### Model Memory

Each model requires different amounts of memory:

| Model | Approximate Memory |
|-------|-------------------|
| whisper-tiny | ~180 MB |
| whisper-base | ~290 MB |
| whisper-small | ~680 MB |
| whisper-medium | ~1.8 GB |
| whisper-large-v3-turbo | ~2.1 GB |
| whisper-large-v3 | ~3.2 GB |
| htdemucs | ~1.2 GB |
| htdemucs_ft | ~1.2 GB |
| clap-htsat-fused | ~650 MB |
| musicgen-small | ~850 MB |
| musicgen-medium | ~2.1 GB |
| parler-tts-mini | ~1.8 GB |
| silero-vad | ~50 MB |
| deepfilternet2 | ~200 MB |

### Audio Memory

Audio arrays also consume memory:

```
Memory = samples × channels × bytes_per_sample

Example (1 minute stereo at 44.1kHz, float32):
44100 × 60 × 2 × 4 = 21.2 MB
```

## Memory Management Techniques

### 1. Choose Appropriate Models

Use smaller models when possible:

```python
import mlx_audio as ma

# Memory-efficient choices
result = ma.transcribe("audio.wav", model="whisper-small")  # vs whisper-large
result = ma.separate("song.mp3", model="htdemucs")  # vs ensemble
result = ma.generate("music", model="musicgen-small")  # vs musicgen-medium
```

### 2. Clear Cache Between Operations

```python
import gc
import mlx.core as mx

def process_with_cleanup(audio_path):
    result = ma.transcribe(audio_path)

    # Force garbage collection
    gc.collect()

    # Clear MLX metal cache
    mx.metal.clear_cache()

    return result.text
```

### 3. Process in Chunks

For long audio files, use chunked processing:

```python
import mlx_audio as ma
import numpy as np

def transcribe_long_audio(audio_path, chunk_seconds=60):
    """Transcribe long audio in chunks."""
    from mlx_audio.primitives import load_audio

    audio, sr = load_audio(audio_path)
    chunk_samples = chunk_seconds * sr

    transcripts = []
    for start in range(0, audio.shape[-1], chunk_samples):
        end = min(start + chunk_samples, audio.shape[-1])
        chunk = audio[:, start:end]

        result = ma.transcribe(chunk, sample_rate=sr)
        transcripts.append(result.text)

        # Clean up between chunks
        del result
        gc.collect()

    return " ".join(transcripts)
```

### 4. Use Streaming for Real-Time

Streaming uses constant memory regardless of audio length:

```python
from mlx_audio.streaming import StreamingPipeline, FileSource, FileSink
from mlx_audio.streaming import HTDemucsStreamProcessor
from mlx_audio.models import HTDemucs

model = HTDemucs.from_pretrained("htdemucs")
pipeline = StreamingPipeline(
    source=FileSource("long_song.mp3"),
    processor=HTDemucsStreamProcessor(model),
    sink=FileSink("vocals.wav", stem_index=3),
)
pipeline.start()
pipeline.wait()
```

### 5. Delete Unused Results

```python
import mlx_audio as ma

# Process and immediately save
result = ma.separate("song.mp3")
result.save("./stems")

# Delete the result object
del result

# Force cleanup
import gc
gc.collect()
```

### 6. Use Lower Precision (Advanced)

For some operations, lower precision reduces memory:

```python
import mlx.core as mx

# Set default to float16 (half precision)
# Note: Not all operations support this
mx.set_default_device(mx.gpu)
```

## Monitoring Memory

### Check Current Usage

```python
import psutil
import os

def print_memory():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    print(f"Memory: {mem:.1f} MB")

print_memory()
result = ma.transcribe("audio.wav")
print_memory()
```

### Profile Memory Usage

```python
import tracemalloc

tracemalloc.start()

result = ma.transcribe("audio.wav")

current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024 / 1024:.1f} MB")
print(f"Peak: {peak / 1024 / 1024:.1f} MB")

tracemalloc.stop()
```

## Memory-Constrained Scenarios

### 8GB RAM Systems

Recommended models:
- whisper-small or smaller
- htdemucs (single model, not ensemble)
- clap-htsat-fused
- silero-vad
- deepfilternet2

### 16GB RAM Systems

Can additionally run:
- whisper-medium
- musicgen-small
- parler-tts-mini

### 32GB+ RAM Systems

Can run all models including:
- whisper-large-v3
- musicgen-medium
- HTDemucs ensemble

## Common Memory Issues

### Out of Memory Errors

```python
# Solution 1: Use smaller model
result = ma.transcribe("audio.wav", model="whisper-small")

# Solution 2: Process in chunks
result = ma.separate("long_song.mp3", segment=3.0)  # Smaller segments

# Solution 3: Clear cache
import gc, mlx.core as mx
gc.collect()
mx.metal.clear_cache()
```

### Memory Leak in Loops

```python
# Problematic
results = []
for audio in audio_files:
    results.append(ma.transcribe(audio))  # Memory grows

# Better
for audio in audio_files:
    result = ma.transcribe(audio)
    save_result(result)  # Save immediately
    del result
    gc.collect()
```

### Multiple Models Loaded

```python
# Problem: Both models in memory
result1 = ma.transcribe("audio.wav")  # Loads Whisper
result2 = ma.separate("song.mp3")      # Loads HTDemucs

# Solution: Process sequentially with cleanup
result1 = ma.transcribe("audio.wav")
save(result1)
del result1
gc.collect()
mx.metal.clear_cache()

result2 = ma.separate("song.mp3")
```

## Best Practices

1. **Start with smaller models** and upgrade only if quality is insufficient
2. **Process files sequentially** rather than loading all into memory
3. **Save results immediately** rather than accumulating in lists
4. **Clear caches** after heavy operations
5. **Use streaming** for long audio or real-time processing
6. **Monitor memory** during development to catch issues early
