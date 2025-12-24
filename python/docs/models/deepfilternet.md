# DeepFilterNet - Audio Enhancement

DeepFilterNet is a neural audio enhancement model that removes background noise and improves audio quality.

## Quick Start

```python
import mlx_audio as ma

# Enhance noisy audio
result = ma.enhance("noisy_recording.wav")
result.save("clean_recording.wav")
```

## Available Models

| Model | Sample Rate | Description | Memory |
|-------|-------------|-------------|--------|
| `deepfilternet2` | 48kHz | High-quality enhancement | ~200 MB |
| `deepfilternet2-16k` | 16kHz | Optimized for speech | ~150 MB |

## Basic Usage

### Simple Enhancement

```python
import mlx_audio as ma

# Enhance and save
result = ma.enhance("noisy_audio.wav")
result.save("clean_audio.wav")

# Or save directly
ma.enhance("noisy_audio.wav", output_file="clean_audio.wav")
```

### Compare Before/After

```python
import mlx_audio as ma

result = ma.enhance("noisy_audio.wav", keep_original=True)

# Access both versions
original, clean = result.before_after
```

## Enhancement Methods

### Neural Enhancement (Default)

Uses DeepFilterNet for high-quality noise reduction:

```python
import mlx_audio as ma

# Neural enhancement (best quality)
result = ma.enhance("audio.wav", method="neural")
```

### Spectral Gating

Simple spectral gating without neural network (faster, no model download):

```python
import mlx_audio as ma

# Spectral gating (no model required)
result = ma.enhance("audio.wav", method="spectral")

# With custom parameters
result = ma.enhance(
    "audio.wav",
    method="spectral",
    threshold_db=-25,      # Noise threshold in dB
    prop_decrease=0.9,     # Amount of noise reduction (0 to 1)
)
```

### Automatic Selection

```python
import mlx_audio as ma

# Auto-selects neural if available, falls back to spectral
result = ma.enhance("audio.wav", method="auto")
```

## Advanced Usage

### Different Models

```python
import mlx_audio as ma

# 48kHz model (default, best for music/general audio)
result = ma.enhance("audio.wav", model="deepfilternet2")

# 16kHz model (optimized for speech)
result = ma.enhance("speech.wav", model="deepfilternet2-16k")
```

### Using Audio Arrays

```python
import mlx_audio as ma
import numpy as np

# Your noisy audio (mono recommended)
audio = np.random.randn(44100 * 10)  # 10 seconds

result = ma.enhance(audio, sample_rate=44100)
clean_audio = result.audio.numpy()
```

## Working with Results

```python
import mlx_audio as ma

result = ma.enhance("noisy.wav")

# Access enhanced audio
audio = result.audio
print(f"Sample rate: {audio.sample_rate}")
print(f"Duration: {result.duration:.2f}s")

# Save to file
result.save("clean.wav")

# Get numpy array
clean_array = audio.numpy()

# With original for comparison
result = ma.enhance("noisy.wav", keep_original=True)
original, enhanced = result.before_after
```

## Use Cases

### Voice Recording Cleanup

```python
import mlx_audio as ma

# Clean up a voice recording with background noise
result = ma.enhance("meeting_recording.wav")
result.save("clean_meeting.wav")
```

### Pre-processing for Transcription

```python
import mlx_audio as ma

# Enhance audio before transcription for better accuracy
enhanced = ma.enhance("noisy_speech.wav")
enhanced.save("temp_clean.wav")

result = ma.transcribe("temp_clean.wav")
```

### Batch Processing

```python
import mlx_audio as ma
from pathlib import Path

input_dir = Path("noisy_files")
output_dir = Path("clean_files")
output_dir.mkdir(exist_ok=True)

for audio_file in input_dir.glob("*.wav"):
    result = ma.enhance(audio_file)
    result.save(output_dir / audio_file.name)
```

## Spectral Gating Parameters

When using `method="spectral"`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold_db` | -20 | Noise threshold in dB below which audio is reduced |
| `prop_decrease` | 0.8 | Amount of noise reduction (0 to 1) |

### Tuning Spectral Gating

```python
import mlx_audio as ma

# Light noise reduction (preserve more audio)
result = ma.enhance(
    "audio.wav",
    method="spectral",
    threshold_db=-30,      # Higher threshold
    prop_decrease=0.5,     # Less aggressive reduction
)

# Aggressive noise reduction
result = ma.enhance(
    "audio.wav",
    method="spectral",
    threshold_db=-15,      # Lower threshold
    prop_decrease=0.95,    # Stronger reduction
)
```

## Performance

### Processing Speed

| Audio Duration | DeepFilterNet | Spectral |
|----------------|---------------|----------|
| 10 seconds | ~0.5s | ~0.1s |
| 1 minute | ~3s | ~0.5s |
| 10 minutes | ~30s | ~5s |

*Benchmarks on M2 Max*

### When to Use Each Method

| Scenario | Recommended Method |
|----------|-------------------|
| High-quality speech enhancement | `neural` (deepfilternet2) |
| Quick processing, many files | `spectral` |
| No model download needed | `spectral` |
| Music or complex audio | `neural` (deepfilternet2) |
| 16kHz speech recordings | `neural` (deepfilternet2-16k) |

## Limitations

- **Mono processing**: Enhancement works on mono audio (stereo is converted)
- **Artifacts**: Aggressive enhancement may introduce slight artifacts
- **Music**: Designed primarily for speech; results on music vary
- **Real-time**: Not optimized for real-time processing (use streaming instead)

## Common Issues

### Enhancement is too aggressive

For spectral method, reduce aggressiveness:

```python
result = ma.enhance("audio.wav", method="spectral", prop_decrease=0.5)
```

### Enhancement is not enough

For spectral method, increase aggressiveness:

```python
result = ma.enhance(
    "audio.wav",
    method="spectral",
    threshold_db=-15,
    prop_decrease=0.95
)
```

Or use neural method for better results:

```python
result = ma.enhance("audio.wav", method="neural")
```

### Wrong sample rate output

DeepFilterNet outputs at its native sample rate. Resample if needed:

```python
from mlx_audio.primitives import resample

result = ma.enhance("audio.wav")
audio = result.audio.array
resampled = resample(audio, result.audio.sample_rate, target_rate)
```

## API Reference

::: mlx_audio.enhance
    options:
      show_root_heading: false
      show_source: false
