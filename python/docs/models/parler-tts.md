# Parler-TTS - Text-to-Speech

Parler-TTS is a text-to-speech model that generates natural-sounding speech with controllable voice characteristics.

## Quick Start

```python
import mlx_audio as ma

# Generate speech
result = ma.speak("Hello, welcome to mlx-audio!")
result.save("greeting.wav")
```

## Available Models

| Model | Description | Memory |
|-------|-------------|--------|
| `parler-tts-mini` | Default model, good quality | ~1.8 GB |

## Basic Usage

### Simple Speech

```python
import mlx_audio as ma

# Convert text to speech
result = ma.speak("The quick brown fox jumps over the lazy dog.")
result.save("output.wav")
```

### Voice Description

Control the voice characteristics with a description:

```python
import mlx_audio as ma

# Warm female voice
result = ma.speak(
    "Welcome to our presentation.",
    description="A warm, friendly female voice speaking clearly"
)

# Professional male voice
result = ma.speak(
    "The quarterly results are in.",
    description="A professional male voice, authoritative and clear"
)
```

### Speed Control

```python
import mlx_audio as ma

# Slower speech
result = ma.speak("Take your time to understand.", speed=0.8)

# Faster speech
result = ma.speak("Quick announcement!", speed=1.3)
```

## Advanced Usage

### Sampling Parameters

```python
import mlx_audio as ma

result = ma.speak(
    "Customized speech synthesis.",
    temperature=0.9,    # Higher = more variation (default: 1.0)
    top_k=50,           # Limit sampling (default: 50)
    seed=42,            # For reproducibility
)
```

### Progress Tracking

```python
import mlx_audio as ma

def on_progress(progress: float):
    print(f"Synthesizing: {progress:.1%}")

result = ma.speak(
    "This is a longer piece of text that will take some time.",
    progress_callback=on_progress
)
```

### Direct File Output

```python
import mlx_audio as ma

# Generate and save in one call
ma.speak(
    "Hello world!",
    output_file="greeting.wav"
)
```

## Voice Descriptions

### Writing Effective Descriptions

Voice descriptions control speaker characteristics:

```python
import mlx_audio as ma

# Describe voice quality
"A clear, articulate voice"
"A soft, gentle voice"
"A deep, resonant voice"

# Describe speaking style
"Speaking slowly and calmly"
"Speaking with enthusiasm and energy"
"Speaking in a measured, professional tone"

# Describe characteristics
"A young female voice"
"An older male voice"
"A friendly, approachable voice"
```

### Example Descriptions

```python
import mlx_audio as ma

# Narrator
result = ma.speak(
    "Once upon a time...",
    description="A warm, storytelling voice, speaking slowly with expression"
)

# Announcer
result = ma.speak(
    "Now arriving at platform 3.",
    description="A clear, professional announcer voice"
)

# Assistant
result = ma.speak(
    "How can I help you today?",
    description="A friendly, helpful female voice"
)

# News reader
result = ma.speak(
    "Breaking news from around the world.",
    description="A professional news anchor voice, clear and authoritative"
)
```

## Working with Results

```python
import mlx_audio as ma

result = ma.speak("Hello!")

# Access audio data
audio = result.audio
print(f"Sample rate: {audio.sample_rate}")
print(f"Duration: {result.duration:.2f}s")

# Save to file
result.save("output.wav")

# Get numpy array for processing
numpy_audio = audio.numpy()
```

## Batch Processing

Generate multiple utterances:

```python
import mlx_audio as ma

texts = [
    "Welcome to the show.",
    "Today we discuss technology.",
    "Thank you for listening."
]

for i, text in enumerate(texts):
    result = ma.speak(text)
    result.save(f"segment_{i}.wav")
```

## Performance Tips

### Speed

- Longer text takes proportionally longer to synthesize
- Use shorter sentences for faster response
- Speed parameter doesn't affect synthesis time, only playback speed

### Memory

Clear cache between large batches:

```python
import gc
import mlx.core as mx

for text in long_text_list:
    result = ma.speak(text)
    result.save(f"output_{i}.wav")
    del result

gc.collect()
mx.metal.clear_cache()
```

## Limitations

- **Voice cloning**: Not supported (uses voice descriptions instead)
- **Emotions**: Limited emotional expression control
- **Languages**: Primarily English (other languages may have reduced quality)
- **Maximum length**: Very long texts may need to be split

## Common Issues

### Unnatural pauses

Add punctuation to control pacing:

```python
# Better pacing with punctuation
ma.speak("Hello! How are you? I hope you're doing well.")
```

### Voice doesn't match description

Try more specific descriptions:

```python
# More specific
description="A calm, soft-spoken female voice, speaking slowly and gently"

# Less effective
description="female voice"
```

### Pronunciation issues

For unusual words, try phonetic spelling or break into syllables in the text.

## API Reference

::: mlx_audio.speak
    options:
      show_root_heading: false
      show_source: false
