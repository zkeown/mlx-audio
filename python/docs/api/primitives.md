# Primitives

Audio DSP primitives for MLX, providing librosa-compatible operations optimized for Apple Silicon.

```python
from mlx_audio import primitives
# or import specific functions
from mlx_audio.primitives import stft, melspectrogram, mfcc
```

## STFT Operations

Core Short-Time Fourier Transform operations.

::: mlx_audio.primitives.stft
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.istft
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.magnitude
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.phase
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.check_nola
    options:
      show_root_heading: true
      show_source: false

## Window Functions

::: mlx_audio.primitives.get_window
    options:
      show_root_heading: true
      show_source: false

## Mel-Scale Operations

::: mlx_audio.primitives.mel_filterbank
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.melspectrogram
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.hz_to_mel
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.mel_to_hz
    options:
      show_root_heading: true
      show_source: false

## Filterbanks

::: mlx_audio.primitives.linear_filterbank
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.bark_filterbank
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.hz_to_bark
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.bark_to_hz
    options:
      show_root_heading: true
      show_source: false

## Spectral Features

::: mlx_audio.primitives.spectral_centroid
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.spectral_bandwidth
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.spectral_rolloff
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.spectral_flatness
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.spectral_contrast
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.zero_crossing_rate
    options:
      show_root_heading: true
      show_source: false

## MFCC

::: mlx_audio.primitives.mfcc
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.delta
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.dct
    options:
      show_root_heading: true
      show_source: false

## Time-Domain Operations

::: mlx_audio.primitives.frame
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.rms
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.preemphasis
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.deemphasis
    options:
      show_root_heading: true
      show_source: false

## Resampling

::: mlx_audio.primitives.resample
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.resample_poly
    options:
      show_root_heading: true
      show_source: false

## Phase Reconstruction

::: mlx_audio.primitives.griffinlim
    options:
      show_root_heading: true
      show_source: false

## Pitch and Periodicity

::: mlx_audio.primitives.autocorrelation
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.pitch_detect_acf
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.yin
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.pyin
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.periodicity
    options:
      show_root_heading: true
      show_source: false

## Onset Detection

::: mlx_audio.primitives.onset_strength
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.onset_strength_multi
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.onset_detect
    options:
      show_root_heading: true
      show_source: false

## Beat Tracking

::: mlx_audio.primitives.tempo
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.beat_track
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.plp
    options:
      show_root_heading: true
      show_source: false

## Spectral Gating

Noise reduction via spectral gating.

::: mlx_audio.primitives.spectral_gate
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.spectral_gate_adaptive
    options:
      show_root_heading: true
      show_source: false

## Decibel Conversions

::: mlx_audio.primitives.power_to_db
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.db_to_power
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.amplitude_to_db
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.primitives.db_to_amplitude
    options:
      show_root_heading: true
      show_source: false
