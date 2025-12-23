# mlx-audio

**Audio machine learning toolkit for Apple Silicon**

mlx-audio provides a comprehensive suite of audio processing tools optimized for Apple Silicon using the [MLX framework](https://github.com/ml-explore/mlx). From speech recognition to music generation, audio enhancement to speaker diarization—all running natively on your Mac.

## Features

- **10 Pre-trained Models** — Whisper, MusicGen, CLAP, HTDemucs, Parler-TTS, and more
- **40+ DSP Primitives** — STFT, mel spectrograms, MFCC, pitch detection, beat tracking
- **Lightning-like Training** — TrainModule and Trainer for custom model development
- **Streaming Support** — Real-time audio processing with sources, sinks, and adapters
- **PyTorch-compatible DataLoader** — Familiar API with MLX optimizations

## Installation

```bash
pip install mlx-audio
```

See the [Installation Guide](installation.md) for detailed instructions.

## Quick Examples

### Transcribe Speech

```python
import mlx_audio as ma

result = ma.transcribe("speech.wav")
print(result.text)
```

### Separate Music Stems

```python
import mlx_audio as ma

result = ma.separate("song.mp3")
result.vocals.save("vocals.wav")
result.drums.save("drums.wav")
```

### Generate Music

```python
import mlx_audio as ma

result = ma.generate("upbeat jazz piano solo", duration=10.0)
result.save("jazz.wav")
```

### Text-to-Speech

```python
import mlx_audio as ma

result = ma.speak("Hello, welcome to mlx-audio!")
result.save("greeting.wav")
```

### Audio Embeddings

```python
import mlx_audio as ma

result = ma.embed(audio="music.wav")
print(result.audio_embedding.shape)
```

### Voice Activity Detection

```python
import mlx_audio as ma

result = ma.detect_speech("recording.wav")
for segment in result.segments:
    print(f"Speech: {segment.start:.2f}s - {segment.end:.2f}s")
```

### Enhance Audio

```python
import mlx_audio as ma

result = ma.enhance("noisy_audio.wav")
result.save("clean_audio.wav")
```

### Speaker Diarization

```python
import mlx_audio as ma

result = ma.diarize("meeting.wav")
for segment in result.segments:
    print(f"Speaker {segment.speaker}: {segment.start:.2f}s - {segment.end:.2f}s")
```

### Audio Classification

```python
import mlx_audio as ma

result = ma.classify("sound.wav", labels=["dog", "cat", "bird"])
print(f"Predicted: {result.label} ({result.confidence:.2%})")
```

### Audio Tagging

```python
import mlx_audio as ma

result = ma.tag("music.wav", tags=["jazz", "piano", "upbeat", "slow"])
print(f"Active tags: {result.active_tags}")
```

## Next Steps

- [Installation](installation.md) — Set up mlx-audio on your system
- [Quickstart](quickstart.md) — Detailed examples for all API functions
- [API Reference](api/functional.md) — Complete API documentation
- [Training Tutorial](tutorials/training-custom-model.md) — Train your own models
