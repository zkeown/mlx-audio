# Whisper - Speech Transcription

Whisper is OpenAI's robust speech recognition model that provides accurate transcription across 99 languages. mlx-audio supports all Whisper model sizes, optimized for Apple Silicon.

## Quick Start

```python
import mlx_audio as ma

# Transcribe audio
result = ma.transcribe("speech.wav")
print(result.text)
```

## Available Models

| Model | Parameters | Speed | Quality | Memory |
|-------|------------|-------|---------|--------|
| `whisper-tiny` | 39M | Fastest | Good | ~40 MB |
| `whisper-base` | 74M | Very Fast | Good | ~75 MB |
| `whisper-small` | 244M | Fast | Better | ~250 MB |
| `whisper-medium` | 769M | Moderate | Very Good | ~750 MB |
| `whisper-large-v3` | 1.5B | Slow | Excellent | ~1.5 GB |
| `whisper-large-v3-turbo` | 809M | Fast | Excellent | ~800 MB |

!!! tip "Recommended Model"
    `whisper-large-v3-turbo` (default) offers the best balance of speed and accuracy. Use `whisper-tiny` or `whisper-small` for real-time applications.

## Basic Usage

### Simple Transcription

```python
import mlx_audio as ma

result = ma.transcribe("audio.wav")
print(result.text)
```

### Specify Language

```python
import mlx_audio as ma

# Force English (faster, no language detection)
result = ma.transcribe("audio.wav", language="en")

# Force Japanese
result = ma.transcribe("japanese.wav", language="ja")
```

### Translation to English

```python
import mlx_audio as ma

# Transcribe non-English audio and translate to English
result = ma.transcribe("french_speech.wav", task="translate")
print(result.text)  # English translation
```

### Save to File

```python
import mlx_audio as ma

# Save as plain text
ma.transcribe("audio.wav", output_file="transcript.txt")

# Save as SRT subtitles
ma.transcribe("video.mp3", output_file="subtitles.srt", output_format="srt")

# Save as WebVTT
ma.transcribe("video.mp3", output_file="subtitles.vtt", output_format="vtt")

# Save as JSON with timestamps
ma.transcribe("audio.wav", output_file="data.json", output_format="json")
```

## Advanced Usage

### Word-Level Timestamps

```python
import mlx_audio as ma

result = ma.transcribe("speech.wav", word_timestamps=True)

for segment in result.segments:
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")

    # Access individual words
    if segment.words:
        for word in segment.words:
            print(f"  {word.word} ({word.start:.2f}s - {word.end:.2f}s)")
```

### Segment-Level Access

```python
import mlx_audio as ma

result = ma.transcribe("speech.wav")

# Iterate over segments
for segment in result.segments:
    print(f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text}")

# Get timing information
print(f"Duration: {result.duration:.2f}s")
print(f"Language: {result.language}")
```

### Progress Tracking

```python
import mlx_audio as ma

def on_progress(progress: float):
    print(f"Transcription progress: {progress:.1%}")

result = ma.transcribe("long_audio.wav", progress_callback=on_progress)
```

### Beam Search

```python
import mlx_audio as ma

# Use beam search for higher accuracy (slower)
result = ma.transcribe("audio.wav", beam_size=5)
```

### Temperature Sampling

```python
import mlx_audio as ma

# Higher temperature for more varied output
result = ma.transcribe("audio.wav", temperature=0.2)

# Temperature 0 is greedy decoding (default, most accurate)
result = ma.transcribe("audio.wav", temperature=0.0)
```

### Using Audio Arrays

```python
import mlx_audio as ma
import numpy as np

# Load your own audio (16kHz mono recommended)
audio = np.random.randn(16000 * 10)  # 10 seconds

result = ma.transcribe(audio, sample_rate=16000)
```

## Working with Results

The `TranscriptionResult` object provides rich access to transcription data:

```python
import mlx_audio as ma

result = ma.transcribe("audio.wav")

# Full text
print(result.text)

# Detected language
print(result.language)  # "en"

# Audio duration
print(result.duration)  # 45.2

# Segments with timing
for seg in result.segments:
    print(f"[{seg.start:.2f} - {seg.end:.2f}] {seg.text}")

# Export to different formats
result.save("transcript.txt")
result.save("subtitles.srt", format="srt")
result.save("data.json", format="json")

# Convert to dictionary
data = result.to_dict()
```

## Language Support

Whisper supports 99 languages. Common language codes:

| Language | Code | Language | Code |
|----------|------|----------|------|
| English | `en` | Spanish | `es` |
| French | `fr` | German | `de` |
| Chinese | `zh` | Japanese | `ja` |
| Korean | `ko` | Portuguese | `pt` |
| Italian | `it` | Russian | `ru` |
| Arabic | `ar` | Hindi | `hi` |

For a complete list, see the [OpenAI Whisper documentation](https://github.com/openai/whisper#available-models-and-languages).

## Model Selection Guide

### When to use each model

- **`whisper-tiny`**: Real-time applications, quick previews, limited memory
- **`whisper-base`**: Good balance for casual transcription
- **`whisper-small`**: Production use with reasonable speed
- **`whisper-large-v3-turbo`**: Best quality with good speed (recommended)
- **`whisper-large-v3`**: Maximum accuracy, when speed is not a concern

### Speed vs Accuracy Tradeoffs

```python
import mlx_audio as ma

# Fast but less accurate
result = ma.transcribe("audio.wav", model="whisper-tiny")

# Best quality (default)
result = ma.transcribe("audio.wav", model="whisper-large-v3-turbo")

# Maximum accuracy
result = ma.transcribe("audio.wav", model="whisper-large-v3", beam_size=5)
```

## Performance Tips

### For Long Audio Files

For audio longer than a few minutes:

```python
import mlx_audio as ma

def on_progress(p):
    print(f"{p:.1%} complete")

result = ma.transcribe("long_podcast.mp3", progress_callback=on_progress)
```

### For Real-Time Applications

```python
import mlx_audio as ma

# Use smaller model for speed
result = ma.transcribe(
    audio_chunk,
    sample_rate=16000,
    model="whisper-tiny",
    language="en",  # Skip language detection
)
```

### Memory Optimization

```python
import mlx_audio as ma

# Use smaller model if memory is constrained
result = ma.transcribe("audio.wav", model="whisper-small")
```

## Common Issues

### Audio Quality

For best results:

- Use 16kHz sample rate (automatic resampling if different)
- Ensure clear audio without excessive background noise
- Consider using `enhance()` for noisy audio first

### Language Detection

If language detection is failing:

```python
# Force the language
result = ma.transcribe("audio.wav", language="en")
```

### Subtitle Timing

For accurate subtitle timing with videos:

```python
result = ma.transcribe(
    "video.mp3",
    word_timestamps=True,  # More precise timing
    output_file="subtitles.srt",
    output_format="srt"
)
```

## API Reference

::: mlx_audio.transcribe
    options:
      show_root_heading: false
      show_source: false
