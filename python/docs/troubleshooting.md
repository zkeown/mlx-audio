# Troubleshooting

Common issues and solutions when using mlx-audio.

## Installation Issues

### MLX Installation Fails

**Problem**: Error installing MLX or mlx-audio.

**Solution**: Ensure you're on Apple Silicon Mac with macOS 13.5+:

```bash
# Check your architecture
uname -m  # Should return "arm64"

# Check macOS version
sw_vers -productVersion  # Should be 13.5 or later

# Install with specific Python version
python3.11 -m pip install mlx-audio
```

### C++ Extensions Not Building

**Problem**: Warning about C++ extensions not available.

**Solution**: This is usually fine - mlx-audio falls back to pure Python/MLX implementations. If you need maximum performance:

```bash
# Install build dependencies
pip install cmake nanobind

# Reinstall mlx-audio
pip install --force-reinstall mlx-audio
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'mlx_audio'`

**Solution**:

```bash
# Verify installation
pip show mlx-audio

# If not installed
pip install mlx-audio

# If using conda, ensure correct environment
conda activate your-env
pip install mlx-audio
```

## Model Loading Issues

### Model Download Fails

**Problem**: Model fails to download from HuggingFace Hub.

**Solution**:

```python
# Check network connectivity
import requests
response = requests.get("https://huggingface.co")
print(response.status_code)  # Should be 200

# Clear cache and retry
import shutil
shutil.rmtree("~/.cache/huggingface", ignore_errors=True)

# Or set custom cache location
import os
os.environ["HF_HOME"] = "/path/to/custom/cache"
```

### Model Not Found

**Problem**: `ModelNotFoundError: Model 'xxx' not found`

**Solution**: Check the model name spelling:

```python
# Correct model names
"htdemucs_ft"           # NOT "htdemucs-ft" or "HT-Demucs"
"whisper-large-v3-turbo"  # Includes hyphens
"clap-htsat-fused"      # Includes hyphens
```

### Out of Memory When Loading Model

**Problem**: `MemoryError` or system becomes unresponsive.

**Solution**:

```python
# Use a smaller model
result = ma.transcribe("audio.wav", model="whisper-small")

# Or clear memory before loading
import gc
import mlx.core as mx
gc.collect()
mx.metal.clear_cache()
```

## Audio Loading Issues

### Unsupported Audio Format

**Problem**: `AudioLoadError: Failed to load audio file`

**Solution**:

```bash
# Install ffmpeg for broader format support
brew install ffmpeg

# Or convert to WAV first
ffmpeg -i input.m4a -ar 44100 -ac 2 output.wav
```

### Sample Rate Mismatch

**Problem**: Unexpected audio quality or errors about sample rate.

**Solution**:

```python
# Explicitly specify sample rate
result = ma.transcribe(audio_array, sample_rate=16000)

# Or let the function handle resampling
result = ma.transcribe("audio.wav")  # Automatic resampling
```

### Empty or Silent Audio

**Problem**: Model produces no output or errors.

**Solution**:

```python
import numpy as np

# Check if audio is valid
audio = ma.load("audio.wav")
print(f"Shape: {audio.shape}")
print(f"Max amplitude: {np.abs(audio).max()}")
print(f"Is silent: {np.abs(audio).max() < 0.001}")

# If silent, check your audio file
```

## Performance Issues

### Slow Processing

**Problem**: Processing takes much longer than expected.

**Solutions**:

1. **Use GPU acceleration**:
```python
import mlx.core as mx
# MLX automatically uses GPU on Apple Silicon
# Ensure Metal is working
print(mx.metal.is_available())  # Should be True
```

2. **Use appropriate model size**:
```python
# Faster models
result = ma.transcribe("audio.wav", model="whisper-tiny")
result = ma.separate("song.mp3", model="htdemucs")  # Not htdemucs_ft
```

3. **Process in smaller chunks**:
```python
# Reduce segment size for streaming
result = ma.separate("long_song.mp3", segment=4.0)
```

### Memory Usage Too High

**Problem**: System runs out of memory during processing.

**Solutions**:

1. **Reduce batch/segment size**:
```python
result = ma.separate("song.mp3", segment=3.0)  # Smaller segments
```

2. **Clear cache between operations**:
```python
import gc
import mlx.core as mx

result = ma.separate("song1.mp3")
del result
gc.collect()
mx.metal.clear_cache()

result = ma.separate("song2.mp3")
```

3. **Use streaming for long files**:
```python
from mlx_audio.streaming import StreamingPipeline, FileSource, FileSink
# Process file in chunks instead of loading entirely
```

### GPU Not Being Used

**Problem**: Processing uses CPU instead of GPU.

**Solution**:

```python
import mlx.core as mx

# Check Metal availability
print(f"Metal available: {mx.metal.is_available()}")
print(f"Default device: {mx.default_device()}")

# If Metal is not available, check:
# 1. You're on Apple Silicon Mac
# 2. macOS 13.5 or later
# 3. MLX is properly installed
```

## Model-Specific Issues

### Whisper

**Transcription has wrong language**:
```python
# Force language detection
result = ma.transcribe("audio.wav", language="en")
```

**Timestamps are inaccurate**:
```python
# Enable word-level timestamps
result = ma.transcribe("audio.wav", word_timestamps=True)
```

**Hallucinations on silence**:
```python
# Use VAD to detect speech first
vad_result = ma.detect_speech("audio.wav")
# Only transcribe speech segments
```

### HTDemucs

**Separation quality is poor**:
```python
# Use fine-tuned model (default)
result = ma.separate("song.mp3", model="htdemucs_ft")

# Or use ensemble for best quality
result = ma.separate("song.mp3", ensemble=True)
```

**Artifacts at segment boundaries**:
```python
# Increase overlap
result = ma.separate("song.mp3", overlap=0.5)
```

### CLAP

**Zero-shot classification not accurate**:
```python
# Use more descriptive labels
labels = [
    "a dog barking loudly",
    "a cat meowing softly",
]
# Instead of just ["dog", "cat"]
```

**Text embeddings fail**:
```python
# Ensure transformers is installed
pip install transformers
```

## Streaming Issues

### Pipeline Stalls

**Problem**: Streaming pipeline stops processing.

**Solution**:

```python
# Check pipeline status
print(f"Running: {pipeline.is_running}")
print(f"Stats: {pipeline.stats}")

# Increase buffer size
from mlx_audio.streaming import StreamingPipeline
pipeline = StreamingPipeline(
    source=source,
    processor=processor,
    sink=sink,
    buffer_size=88200,  # 2 seconds at 44.1kHz
)
```

### Real-Time Performance

**Problem**: Processing can't keep up with real-time.

**Solution**:

```python
# Check real-time factor
stats = pipeline.stats
print(f"Real-time factor: {stats.realtime_factor}")
# Should be > 1.0 for real-time performance

# If < 1.0, use smaller model or increase chunk size
```

### Microphone Access Denied

**Problem**: `PermissionError` when using MicrophoneSource.

**Solution**:
1. Go to System Preferences > Privacy & Security > Microphone
2. Enable microphone access for Terminal (or your IDE)
3. Restart your application

## Common Error Messages

### `ConfigurationError: At least one of audio or text must be provided`

You called `embed()` without any input:
```python
# Wrong
result = ma.embed()

# Correct
result = ma.embed(audio="audio.wav")
result = ma.embed(text="description")
```

### `TokenizationError: transformers is required`

Install the transformers library:
```bash
pip install transformers
```

### `AudioLoadError: Failed to load audio`

The audio file path is invalid or format is unsupported:
```python
from pathlib import Path

# Check file exists
path = Path("audio.wav")
print(f"Exists: {path.exists()}")
print(f"Size: {path.stat().st_size if path.exists() else 'N/A'}")
```

## Getting Help

If you're still having issues:

1. **Check the GitHub issues**: [github.com/zkeown/mlx-audio/issues](https://github.com/zkeown/mlx-audio/issues)

2. **Report a bug**: Include:
   - mlx-audio version (`pip show mlx-audio`)
   - Python version (`python --version`)
   - macOS version (`sw_vers`)
   - Full error traceback
   - Minimal code to reproduce

3. **Ask questions**: Open a discussion on GitHub
