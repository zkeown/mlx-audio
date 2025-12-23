# mlx-audio (Python)

Python package for mlx-audio. See the [main README](../README.md) for full documentation.

## Installation

```bash
pip install git+https://github.com/zkeown/mlx-audio.git#subdirectory=python
```

## Development Setup

```bash
git clone https://github.com/zkeown/mlx-audio
cd mlx-audio/python
pip install -e ".[dev]"
```

## Quick Example

```python
import mlx_audio

# Transcribe speech
result = mlx_audio.transcribe("speech.wav")
print(result.text)

# Separate music stems
stems = mlx_audio.separate("song.mp3")
stems.vocals.save("vocals.wav")

# Generate music
audio = mlx_audio.generate("jazz piano, upbeat mood", duration=10.0)
audio.save("output.wav")
```

## Documentation

- [Full Documentation](docs/index.md)
- [API Reference](docs/api/functional.md)
- [Training Tutorial](docs/tutorials/training-custom-model.md)
- [Contributing Guide](docs/contributing.md)

## License

MIT — see [LICENSE](../LICENSE) for details.
