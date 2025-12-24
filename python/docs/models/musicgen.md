# MusicGen - Music Generation

MusicGen is a text-to-music generation model from Meta AI that creates high-quality music from text descriptions.

## Quick Start

```python
import mlx_audio as ma

# Generate music from a text prompt
result = ma.generate("upbeat jazz piano solo")
result.save("jazz.wav")
```

## Available Models

| Model | Parameters | Speed | Quality | Memory |
|-------|------------|-------|---------|--------|
| `musicgen-small` | 300M | Fast | Good | ~850 MB |
| `musicgen-medium` | 1.5B | Moderate | Better | ~2.1 GB |

## Basic Usage

### Simple Generation

```python
import mlx_audio as ma

# Generate 10 seconds of music
result = ma.generate("electronic dance music with heavy bass")
result.save("output.wav")
```

### Custom Duration

```python
import mlx_audio as ma

# Generate up to 30 seconds
result = ma.generate("calm ambient piano", duration=20.0)
```

!!! note "Duration Limit"
    MusicGen supports a maximum duration of 30 seconds per generation.

### Save to File

```python
import mlx_audio as ma

# Generate and save directly
ma.generate("rock guitar solo", output_file="guitar.wav")
```

## Advanced Usage

### Sampling Parameters

Control the randomness and creativity of generation:

```python
import mlx_audio as ma

result = ma.generate(
    "orchestral movie soundtrack",
    temperature=0.9,    # Higher = more creative (default: 1.0)
    top_k=250,          # Limit sampling to top k tokens (default: 250)
    top_p=0.95,         # Nucleus sampling threshold (default: 0.0)
    cfg_scale=3.0,      # Classifier-free guidance (default: 3.0)
)
```

### Reproducible Generation

```python
import mlx_audio as ma

# Use a seed for reproducible results
result = ma.generate("funk bass groove", seed=42)

# Same seed = same output
result2 = ma.generate("funk bass groove", seed=42)
# result and result2 are identical
```

### Progress Tracking

```python
import mlx_audio as ma

def on_progress(progress: float):
    print(f"Generating: {progress:.1%}")

result = ma.generate(
    "cinematic epic music",
    duration=20.0,
    progress_callback=on_progress
)
```

## Prompt Engineering

### Effective Prompts

Good prompts are specific and descriptive:

```python
# Good prompts
"upbeat jazz piano trio with walking bass and brushed drums"
"melancholic acoustic guitar with fingerpicking, slow tempo"
"energetic electronic dance music, 128 BPM, deep bass drops"
"peaceful ambient soundscape with soft synth pads"

# Less effective (too vague)
"music"
"good song"
"something nice"
```

### Genre Examples

```python
import mlx_audio as ma

# Jazz
ma.generate("smooth jazz saxophone solo, mellow mood")

# Electronic
ma.generate("deep house music, four on the floor beat, synth leads")

# Classical
ma.generate("romantic piano piece in the style of Chopin")

# Rock
ma.generate("heavy metal guitar riff with double bass drums")

# Ambient
ma.generate("ethereal ambient textures, reverb, space sounds")
```

### Mood and Atmosphere

```python
import mlx_audio as ma

# Happy/Upbeat
ma.generate("cheerful pop melody, bright and energetic")

# Sad/Melancholic
ma.generate("melancholic minor key piano, slow and emotional")

# Intense/Dramatic
ma.generate("epic orchestral build-up, intense percussion")

# Relaxing/Calm
ma.generate("gentle acoustic lullaby, soft and soothing")
```

## Working with Results

The `GenerationResult` object provides access to generated audio:

```python
import mlx_audio as ma

result = ma.generate("funky bass groove")

# Access audio data
audio = result.audio          # AudioData object
print(audio.shape)            # [channels, samples]
print(audio.sample_rate)      # 32000

# Save to file
result.save("output.wav")

# Get numpy array
numpy_audio = audio.numpy()

# Get duration
print(f"Duration: {result.duration:.2f}s")
```

## Performance Tips

### Model Selection

```python
import mlx_audio as ma

# Faster generation (lower quality)
result = ma.generate("pop music", model="musicgen-small")

# Better quality (slower)
result = ma.generate("pop music", model="musicgen-medium")
```

### Memory Management

For long sessions with multiple generations:

```python
import gc
import mlx.core as mx

# After each generation
result = ma.generate("music prompt")
result.save("output.wav")
del result
gc.collect()
mx.metal.clear_cache()
```

### Generation Speed

Approximate generation times on M2 Max:

| Model | 10 seconds | 20 seconds | 30 seconds |
|-------|------------|------------|------------|
| musicgen-small | ~15s | ~30s | ~45s |
| musicgen-medium | ~32s | ~65s | ~95s |

## Limitations

- **Maximum duration**: 30 seconds per generation
- **Sample rate**: Output is 32kHz
- **Mono output**: Single channel audio
- **Text encoder**: Requires T5 model (downloaded automatically)

## Common Issues

### Generation is slow

Use the smaller model for faster results:

```python
result = ma.generate("music", model="musicgen-small")
```

### Output doesn't match prompt

Try adjusting guidance:

```python
# Higher cfg_scale = more prompt adherence
result = ma.generate("specific style", cfg_scale=5.0)
```

### Repetitive output

Increase temperature for more variety:

```python
result = ma.generate("music", temperature=1.2, top_p=0.95)
```

## API Reference

::: mlx_audio.generate
    options:
      show_root_heading: false
      show_source: false
