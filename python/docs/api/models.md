# Models

mlx-audio provides pre-trained models across 10 task categories, all optimized for Apple Silicon.

## Model Registry

Models are managed through a central registry with lazy loading and automatic HuggingFace Hub integration.

```python
from mlx_audio.hub.registry import ModelRegistry, TaskType

registry = ModelRegistry.get()

# List all models
all_models = registry.list_models()

# List models for a specific task
transcription_models = registry.list_models(TaskType.TRANSCRIPTION)

# Get default model for a task
default = registry.get_default_for_task(TaskType.TRANSCRIPTION)
print(default.name)  # "whisper-large-v3-turbo"
```

## Available Models by Task

### Transcription (Speech-to-Text)

| Model | Default | Capabilities |
|-------|---------|--------------|
| `whisper-large-v3-turbo` | Yes | streaming, batched, timestamps, multilingual |
| `whisper-large-v3` | | streaming, batched, timestamps, multilingual |
| `whisper-large-v2` | | streaming, batched, timestamps, multilingual |
| `whisper-medium` | | streaming, batched, timestamps, multilingual |
| `whisper-small` | | streaming, batched, timestamps, multilingual |
| `whisper-base` | | streaming, batched, timestamps, multilingual |
| `whisper-tiny` | | streaming, batched, timestamps, multilingual |

```python
import mlx_audio as ma

# Use default model
result = ma.transcribe("speech.wav")

# Specify a smaller model
result = ma.transcribe("speech.wav", model="whisper-small")
```

### Separation (Source Separation)

| Model | Default | Capabilities |
|-------|---------|--------------|
| `htdemucs` | Yes | streaming, batched |
| `htdemucs_ft` | | streaming, batched |

```python
import mlx_audio as ma

# Use default model (htdemucs)
result = ma.separate("song.mp3")

# Use fine-tuned version
result = ma.separate("song.mp3", model="htdemucs_ft")
```

### Generation (Audio Synthesis)

| Model | Default | Capabilities |
|-------|---------|--------------|
| `musicgen-medium` | Yes | variable_length, cfg, streaming |
| `musicgen-small` | | variable_length, cfg, streaming |
| `musicgen-large` | | variable_length, cfg, streaming |
| `musicgen-melody` | | variable_length, cfg, melody_conditioning |

```python
import mlx_audio as ma

# Use default model
result = ma.generate("jazz piano solo")

# Use smaller model for faster generation
result = ma.generate("ambient music", model="musicgen-small")
```

### Speech (Text-to-Speech)

| Model | Default | Capabilities |
|-------|---------|--------------|
| `parler-tts-mini` | Yes | voice_description, variable_length |
| `parler-tts-large` | | voice_description, variable_length |

```python
import mlx_audio as ma

# Use default model
result = ma.speak("Hello, world!")

# Use larger model for higher quality
result = ma.speak("Hello, world!", model="parler-tts-large")
```

### Embedding (Audio-Text)

| Model | Default | Capabilities |
|-------|---------|--------------|
| `clap-htsat-fused` | Yes | variable_length, batched, text_encoding |
| `clap-htsat-unfused` | | batched, text_encoding |

```python
import mlx_audio as ma

# Audio embedding
result = ma.embed(audio="sound.wav")

# Zero-shot classification
result = ma.embed(
    audio="sound.wav",
    text=["dog", "cat", "bird"],
    return_similarity=True,
)
```

### VAD (Voice Activity Detection)

| Model | Default | Capabilities |
|-------|---------|--------------|
| `silero-vad` | Yes | streaming, real_time |
| `silero-vad-8k` | | streaming, real_time |

```python
import mlx_audio as ma

# Use default model
result = ma.detect_speech("recording.wav")

# Use 8kHz model for telephony audio
result = ma.detect_speech("phone.wav", model="silero-vad-8k")
```

### Enhancement (Noise Reduction)

| Model | Default | Capabilities |
|-------|---------|--------------|
| `deepfilternet2` | Yes | streaming, real_time |
| `deepfilternet2-16k` | | streaming, real_time |

```python
import mlx_audio as ma

# Use default model (48kHz)
result = ma.enhance("noisy.wav")

# Use 16kHz model for speech
result = ma.enhance("speech.wav", model="deepfilternet2-16k")
```

### Diarization (Speaker Identification)

| Model | Default | Capabilities |
|-------|---------|--------------|
| `ecapa-tdnn` | Yes | batched, speaker_embeddings |

```python
import mlx_audio as ma

result = ma.diarize("meeting.wav")
```

### Classification & Tagging

Classification and tagging use CLAP models for zero-shot inference:

| Model | Default | Capabilities |
|-------|---------|--------------|
| `clap-htsat-fused` | Yes | variable_length, batched, text_encoding |

```python
import mlx_audio as ma

# Zero-shot classification
result = ma.classify("sound.wav", labels=["speech", "music", "noise"])

# Zero-shot tagging
result = ma.tag("music.wav", tags=["piano", "drums", "vocals"])
```

## Custom Models

### Using HuggingFace Repos

You can use any compatible HuggingFace model by passing the repo ID:

```python
import mlx_audio as ma

# Use a custom Whisper model
result = ma.transcribe("audio.wav", model="mlx-community/whisper-medium.en")
```

### Registering Custom Models

Register your own models with the registry:

```python
from mlx_audio.hub.registry import register_model, TaskType

@register_model(
    name="my-transcriber",
    task=TaskType.TRANSCRIPTION,
    default_repo="username/my-model",
    capabilities=["batched"],
)
class MyTranscriber:
    def __init__(self, repo_id: str):
        # Load weights from repo_id
        pass

    def __call__(self, audio):
        # Perform transcription
        pass
```

## Model Caching

Models are automatically cached after first download:

```python
import os

# Set custom cache directory
os.environ["HF_HOME"] = "/path/to/cache"

# Models will be downloaded to /path/to/cache/hub/
```

## Model Capabilities

Each model has associated capabilities that indicate its features:

| Capability | Description |
|------------|-------------|
| `streaming` | Supports streaming/chunked processing |
| `batched` | Supports batch processing |
| `real_time` | Suitable for real-time applications |
| `variable_length` | Handles variable-length inputs |
| `timestamps` | Provides timing information |
| `multilingual` | Supports multiple languages |
| `text_encoding` | Can encode text inputs |
| `voice_description` | Accepts voice descriptions |
| `cfg` | Supports classifier-free guidance |
| `melody_conditioning` | Accepts melody input |
