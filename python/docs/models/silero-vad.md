# SileroVAD - Voice Activity Detection

SileroVAD is a lightweight, fast voice activity detection model that identifies speech segments in audio.

## Quick Start

```python
import mlx_audio as ma

# Detect speech segments
result = ma.detect_speech("recording.wav")

for segment in result.segments:
    print(f"Speech: {segment.start:.2f}s - {segment.end:.2f}s")
```

## Available Models

| Model | Description | Memory |
|-------|-------------|--------|
| `silero-vad` | Default VAD model | ~50 MB |

## Basic Usage

### Detect Speech Segments

```python
import mlx_audio as ma

result = ma.detect_speech("audio.wav")

print(f"Found {len(result.segments)} speech segments")

for segment in result.segments:
    print(f"  {segment.start:.2f}s - {segment.end:.2f}s")
```

### Adjust Sensitivity

```python
import mlx_audio as ma

# More sensitive (detect quieter speech)
result = ma.detect_speech("audio.wav", threshold=0.3)

# Less sensitive (reduce false positives)
result = ma.detect_speech("audio.wav", threshold=0.7)
```

### Segment Duration Filters

```python
import mlx_audio as ma

result = ma.detect_speech(
    "audio.wav",
    min_speech_duration=0.5,    # Ignore segments shorter than 0.5s
    min_silence_duration=0.3,   # Merge segments with gaps < 0.3s
)
```

## Advanced Usage

### Access Probabilities

Get frame-by-frame speech probabilities:

```python
import mlx_audio as ma

result = ma.detect_speech("audio.wav", return_probabilities=True)

# Speech probability for each frame
print(f"Probabilities shape: {result.probabilities.shape}")

# Average speech probability
avg_prob = result.probabilities.mean()
print(f"Average speech probability: {avg_prob:.2%}")
```

### Export Results

```python
import mlx_audio as ma

# Save as JSON
ma.detect_speech("audio.wav", output_file="vad.json", output_format="json")

# Save as plain text
ma.detect_speech("audio.wav", output_file="vad.txt", output_format="txt")

# Save as Audacity labels
ma.detect_speech("audio.wav", output_file="vad.txt", output_format="audacity")
```

### Progress Tracking

```python
import mlx_audio as ma

def on_progress(progress: float):
    print(f"Processing: {progress:.1%}")

result = ma.detect_speech("long_audio.wav", progress_callback=on_progress)
```

### Using Audio Arrays

```python
import mlx_audio as ma
import numpy as np

# 16kHz mono audio (VAD's native format)
audio = np.random.randn(16000 * 10)  # 10 seconds

result = ma.detect_speech(audio, sample_rate=16000)
```

## Working with Results

### VADResult Object

```python
import mlx_audio as ma

result = ma.detect_speech("audio.wav")

# List of speech segments
for segment in result.segments:
    print(f"Start: {segment.start}")
    print(f"End: {segment.end}")
    print(f"Duration: {segment.duration}")

# Summary statistics
print(f"Total speech: {result.total_speech_duration:.2f}s")
print(f"Number of segments: {len(result.segments)}")
```

### Extract Speech Audio

```python
import mlx_audio as ma

# Load audio
result = ma.detect_speech("audio.wav")

# Get only speech portions (concatenated)
speech_audio = result.get_speech_audio(audio_array)
```

## Use Cases

### Pre-processing for Transcription

Use VAD to identify speech segments before transcription:

```python
import mlx_audio as ma

# First, detect speech
vad_result = ma.detect_speech("meeting.wav")

# Then transcribe only speech segments
for segment in vad_result.segments:
    # Extract segment audio and transcribe
    print(f"Transcribing {segment.start:.1f}s - {segment.end:.1f}s")
```

### Skip Silence in Long Recordings

```python
import mlx_audio as ma

result = ma.detect_speech("podcast.wav")

# Calculate speech ratio
total_duration = audio_duration  # Your audio duration
speech_ratio = result.total_speech_duration / total_duration

print(f"Speech content: {speech_ratio:.1%}")
print(f"You can skip {(1 - speech_ratio) * 100:.1f}% silence")
```

### Speaker Turn Detection

Rough approximation of speaker turns:

```python
import mlx_audio as ma

result = ma.detect_speech(
    "conversation.wav",
    min_silence_duration=0.5  # Gaps > 0.5s likely indicate speaker change
)

print(f"Approximate speaker turns: {len(result.segments)}")
```

## Performance

### Speed

SileroVAD is very fast:

| Audio Duration | Processing Time |
|----------------|-----------------|
| 1 minute | ~50ms |
| 10 minutes | ~500ms |
| 1 hour | ~3s |

### Memory

Minimal memory footprint (~50 MB for the model).

## Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 0.5 | Speech probability threshold (0.0 to 1.0) |
| `min_speech_duration` | 0.25 | Minimum speech segment length (seconds) |
| `min_silence_duration` | 0.1 | Minimum silence to split segments (seconds) |
| `return_probabilities` | False | Include per-frame probabilities |

### Threshold Guidelines

| Threshold | Use Case |
|-----------|----------|
| 0.3 | High recall (catch all speech, some false positives) |
| 0.5 | Balanced (default) |
| 0.7 | High precision (fewer false positives, may miss quiet speech) |
| 0.9 | Very strict (only clear, loud speech) |

## Common Issues

### Too many segments

Increase `min_speech_duration` to merge short segments:

```python
result = ma.detect_speech("audio.wav", min_speech_duration=0.5)
```

### Missing speech

Lower the threshold:

```python
result = ma.detect_speech("audio.wav", threshold=0.3)
```

### False positives on background noise

Raise the threshold:

```python
result = ma.detect_speech("audio.wav", threshold=0.7)
```

## API Reference

::: mlx_audio.detect_speech
    options:
      show_root_heading: false
      show_source: false
