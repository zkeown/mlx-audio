# mlx-audio Examples

This directory contains standalone example scripts demonstrating mlx-audio capabilities.

## Quick Start Examples

| Script | Description | Time to Run |
|--------|-------------|-------------|
| [01_transcribe.py](01_transcribe.py) | Speech-to-text transcription | ~30s |
| [02_separate.py](02_separate.py) | Music source separation | ~1min |
| [03_embed.py](03_embed.py) | Audio embeddings & zero-shot classification | ~10s |
| [04_generate.py](04_generate.py) | Text-to-music generation | ~2min |
| [05_speak.py](05_speak.py) | Text-to-speech synthesis | ~30s |
| [06_detect_speech.py](06_detect_speech.py) | Voice activity detection | ~5s |
| [07_enhance.py](07_enhance.py) | Audio denoising | ~10s |
| [08_diarize.py](08_diarize.py) | Speaker diarization | ~30s |

## Running Examples

Most examples can run with default settings:

```bash
cd examples
python 01_transcribe.py
```

Some examples require audio files. You can use your own or download samples:

```bash
# Download sample audio files
python download_samples.py
```

## License Information

Some models have commercial use restrictions. Check before deploying:

```python
import mlx_audio

# List models safe for commercial use
print(mlx_audio.list_commercial_safe_models())

# Check a specific model
print(mlx_audio.is_commercial_safe("musicgen-medium"))  # False (non-commercial only)
print(mlx_audio.is_commercial_safe("whisper-large-v3-turbo"))  # True
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- mlx-audio installed (`pip install -e ../python[dev]`)
