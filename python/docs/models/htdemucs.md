# HTDemucs - Source Separation

HTDemucs is a state-of-the-art audio source separation model that splits audio into individual stems (vocals, drums, bass, and other instruments). It's based on the [Hybrid Transformer Demucs](https://github.com/facebookresearch/demucs) architecture from Meta AI.

## Quick Start

```python
import mlx_audio as ma

# Separate a song into stems
result = ma.separate("song.mp3")

# Access individual stems
print(result.vocals.shape)   # Vocal track
print(result.drums.shape)    # Drum track
print(result.bass.shape)     # Bass track
print(result.other.shape)    # Other instruments
```

## Available Models

| Model | Stems | Description |
|-------|-------|-------------|
| `htdemucs_ft` | 4 | Fine-tuned model (default, best quality for 4-stem separation) |
| `htdemucs` | 4 | Base model (faster, slightly lower quality) |
| `htdemucs_6s` | 6 | Extended model with guitar and piano stems |

### Stem Types

**4-stem models** (`htdemucs`, `htdemucs_ft`):

- `vocals` - Singing and spoken voice
- `drums` - Percussion instruments
- `bass` - Bass guitar, upright bass
- `other` - All other instruments

**6-stem models** (`htdemucs_6s`):

- `vocals`, `drums`, `bass` (same as above)
- `guitar` - Electric and acoustic guitars
- `piano` - Piano and keyboards
- `other` - Remaining instruments

## Basic Usage

### Separate and Save

```python
import mlx_audio as ma

# Separate and automatically save stems to a directory
ma.separate("song.mp3", output_dir="./stems")
# Creates: ./stems/vocals.wav, ./stems/drums.wav, etc.
```

### Select Specific Stems

```python
import mlx_audio as ma

# Only extract vocals and drums (faster)
result = ma.separate("song.mp3", stems=["vocals", "drums"])
```

### Manual Saving

```python
import mlx_audio as ma

result = ma.separate("song.mp3")

# Save individual stems
result.vocals.save("vocals.wav")
result.drums.save("drums.wav")

# Or save all at once
result.save("./output_folder")
```

## Advanced Usage

### Ensemble Mode

Use an ensemble of 4 specialized models for higher quality (~3dB SDR improvement):

```python
import mlx_audio as ma

# Enable ensemble mode for best quality
result = ma.separate("song.mp3", ensemble=True)
```

!!! note "Ensemble Requirements"
    Ensemble mode requires the `htdemucs_ft_bag` models to be downloaded. This uses approximately 4x the memory and processing time of a single model.

### Chunked Processing

For long audio files or limited memory, adjust the segment and overlap parameters:

```python
import mlx_audio as ma

# Process in smaller chunks (uses less memory)
result = ma.separate(
    "long_song.mp3",
    segment=4.0,      # Process 4 seconds at a time (default: 6.0)
    overlap=0.25,     # 25% overlap between segments (default: 0.25)
)
```

### Progress Tracking

```python
import mlx_audio as ma

def on_progress(current, total):
    print(f"Processing segment {current}/{total}")

result = ma.separate("song.mp3", progress_callback=on_progress)
```

### Using Audio Arrays

```python
import mlx_audio as ma
import numpy as np

# Load your own audio (stereo, 44.1kHz recommended)
audio = np.random.randn(2, 44100 * 10)  # 10 seconds

result = ma.separate(audio, sample_rate=44100)
```

### 6-Stem Separation

```python
import mlx_audio as ma

# Use 6-stem model for guitar and piano
result = ma.separate("song.mp3", model="htdemucs_6s")

print(result.guitar.shape)
print(result.piano.shape)
```

## Working with Results

The `SeparationResult` object provides convenient access to stems:

```python
import mlx_audio as ma

result = ma.separate("song.mp3")

# Access stems as attributes
vocals = result.vocals        # AudioData object
drums = result.drums

# Get audio array and sample rate
audio_array = vocals.array    # mlx.array [C, T]
sample_rate = vocals.sample_rate  # int (44100)

# Convert to numpy
numpy_array = vocals.numpy()

# Get available stems
print(result.stem_names)  # ['vocals', 'drums', 'bass', 'other']

# Iterate over stems
for name, audio_data in result.items():
    print(f"{name}: {audio_data.shape}")
```

## Performance Tips

### Memory Usage

- **Segment size**: Smaller `segment` values use less memory but may produce more artifacts at boundaries
- **Stem selection**: Only request the stems you need with the `stems` parameter
- **Model choice**: `htdemucs` is faster than `htdemucs_ft` with slightly lower quality

### Processing Speed

| Audio Length | htdemucs | htdemucs_ft | Ensemble |
|--------------|----------|-------------|----------|
| 30 seconds   | ~2s      | ~2s         | ~8s      |
| 3 minutes    | ~10s     | ~12s        | ~45s     |
| 10 minutes   | ~35s     | ~40s        | ~150s    |

*Benchmarks on M2 Max, 32GB RAM*

## Technical Details

- **Sample rate**: 44,100 Hz (audio is automatically resampled)
- **Channels**: Stereo (mono is converted to stereo)
- **Output format**: Same sample rate as model (44,100 Hz)

## Common Issues

### Out of Memory

If you encounter memory issues with long audio files:

```python
# Reduce segment size
result = ma.separate("long_song.mp3", segment=3.0)
```

### Stem Quality

For best quality on difficult material:

1. Use `htdemucs_ft` (default)
2. Enable ensemble mode if quality is critical
3. Ensure audio is 44.1kHz stereo

## API Reference

::: mlx_audio.separate
    options:
      show_root_heading: false
      show_source: false
