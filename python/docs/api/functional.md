# High-Level API

The high-level API provides 10 one-liner functions for common audio tasks. All functions accept file paths, numpy arrays, or MLX arrays as input.

## Audio Processing

### separate

::: mlx_audio.separate
    options:
      show_root_heading: true
      show_source: false

### transcribe

::: mlx_audio.transcribe
    options:
      show_root_heading: true
      show_source: false

### enhance

::: mlx_audio.enhance
    options:
      show_root_heading: true
      show_source: false

## Audio Generation

### generate

::: mlx_audio.generate
    options:
      show_root_heading: true
      show_source: false

### speak

::: mlx_audio.speak
    options:
      show_root_heading: true
      show_source: false

## Audio Analysis

### embed

::: mlx_audio.embed
    options:
      show_root_heading: true
      show_source: false

### detect_speech

::: mlx_audio.detect_speech
    options:
      show_root_heading: true
      show_source: false

### diarize

::: mlx_audio.diarize
    options:
      show_root_heading: true
      show_source: false

### classify

::: mlx_audio.classify
    options:
      show_root_heading: true
      show_source: false

### tag

::: mlx_audio.tag
    options:
      show_root_heading: true
      show_source: false

## Re-exported Primitives

These commonly used primitives are available directly from `mlx_audio`:

```python
from mlx_audio import stft, istft, melspectrogram, mfcc, resample, griffinlim
```

See [Primitives](primitives.md) for full documentation.

## Re-exported Training Classes

These training utilities are available directly from `mlx_audio`:

```python
from mlx_audio import TrainModule, Trainer, OptimizerConfig, DataLoader, Dataset
```

See [Training](training.md) for full documentation.
