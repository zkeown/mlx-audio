# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mlx-audio is an audio machine learning toolkit for Apple Silicon using the MLX framework. It provides audio DSP primitives (STFT, mel spectrogram, MFCC), a PyTorch-compatible DataLoader, a Lightning-like training framework, and pre-built models (HTDemucs, CLAP, MusicGen, Whisper, EnCodec).

**Requirements:** macOS + Apple Silicon, Python 3.11+, MLX 0.30.0+

## Build & Development Commands

### Swift

```bash
# Build Swift package
cd swift
swift build

# Run Swift tests (IMPORTANT: use the test script, not swift test directly)
./swift/test.sh

# Run specific Swift test
./swift/test.sh --filter BanquetParityTests
```

### Python

```bash
# Install in development mode (from python/ directory)
cd python
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_imports.py

# Run tests with coverage
pytest tests/ --cov=mlx_audio --cov-report=html

# Run tests by marker
pytest -m "not slow"        # Skip slow tests
pytest -m parity            # Parity tests against reference implementations
pytest -m integration       # Integration tests only

# Lint and format
ruff check python/mlx_audio
ruff format python/mlx_audio

# Type checking
mypy python/mlx_audio
```

## Architecture

### Package Structure (python/mlx_audio/)

| Module | Purpose |
|--------|---------|
| `primitives/` | Audio DSP operations with optional C++ acceleration |
| `data/` | DataLoader and dataset classes (PyTorch-compatible API) |
| `train/` | Lightning-like training framework (TrainModule, Trainer) |
| `models/` | Pre-built models (demucs/, clap/, musicgen/, whisper/, encodec/) |
| `functional/` | High-level task APIs (separate, transcribe, generate, embed) |
| `hub/` | Model registry and caching system |
| `streaming/` | Real-time audio I/O (sources, sinks, adapters) |
| `types/` | Result types (SeparationResult, TranscriptionResult, etc.) |

### C++ Extensions (python/csrc/)

C++/Metal primitives for performance-critical operations. Uses CMake + nanobind for building. Key operations: `overlap_add`, `frame_signal`, `pad_signal`, `mel_filterbank`.

The Python code gracefully degrades if C++ extensions are unavailable—check `HAS_CPP_EXT` flag in primitives.

### Public API

Four high-level functions in `mlx_audio/__init__.py`:
- `separate(audio, model="htdemucs_ft")` → SeparationResult
- `transcribe(audio, model="whisper-large-v3-turbo")` → TranscriptionResult
- `generate(prompt, model="musicgen-medium")` → GenerationResult
- `embed(audio=None, text=None, model="clap-htsat-fused")` → EmbeddingResult

### Model Loading Pattern

Models use lazy-loading with HuggingFace Hub caching:
```python
from mlx_audio.hub.cache import get_cache
cache = get_cache()
model = cache.get_model("htdemucs_ft", HTDemucs)
```

Each model directory contains a `convert.py` for PyTorch → MLX weight conversion.

## Code Conventions

- Line length: 100 characters (ruff-enforced)
- Type hints required (mypy-checked)
- Parity tests verify correctness against librosa/torchaudio reference implementations
- Property-based tests use Hypothesis for robustness
