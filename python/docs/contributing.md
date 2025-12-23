# Contributing

Thank you for your interest in contributing to mlx-audio! This guide will help you get started.

## Development Setup

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11 or higher
- Git

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/zkeown/mlx-audio.git
cd mlx-audio/python

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install in development mode with all extras
pip install -e ".[dev,docs]"
```

### Verify Installation

```bash
# Run tests
pytest tests/

# Check linting
ruff check mlx_audio

# Check types
mypy mlx_audio
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_imports.py

# Run tests with coverage
pytest tests/ --cov=mlx_audio --cov-report=html

# Run tests by marker
pytest -m "not slow"        # Skip slow tests
pytest -m parity            # Parity tests against reference implementations
pytest -m integration       # Integration tests only

# Run tests in parallel
pytest tests/ -n auto
```

## Code Style

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting, and [mypy](http://mypy-lang.org/) for type checking.

### Formatting

```bash
# Check formatting
ruff format --check mlx_audio

# Auto-format code
ruff format mlx_audio
```

### Linting

```bash
# Check for issues
ruff check mlx_audio

# Auto-fix issues
ruff check --fix mlx_audio
```

### Type Checking

```bash
# Run mypy
mypy mlx_audio
```

### Style Guidelines

- **Line length**: 100 characters
- **Type hints**: Required for all public functions
- **Docstrings**: Google style, required for public APIs
- **Imports**: Sorted by ruff (isort-compatible)

Example function with proper style:

```python
def process_audio(
    audio: mx.array,
    sample_rate: int = 16000,
    normalize: bool = True,
) -> mx.array:
    """Process audio signal for model input.

    Args:
        audio: Input audio array of shape [samples] or [batch, samples].
        sample_rate: Audio sample rate in Hz.
        normalize: Whether to normalize amplitude to [-1, 1].

    Returns:
        Processed audio array ready for model input.

    Raises:
        ValueError: If sample_rate is not positive.

    Example:
        >>> audio = mx.random.normal((16000,))
        >>> processed = process_audio(audio, sample_rate=16000)
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    if normalize:
        audio = audio / mx.max(mx.abs(audio) + 1e-8)

    return audio
```

## Project Structure

```
python/
├── mlx_audio/
│   ├── __init__.py          # Public API
│   ├── primitives/           # DSP operations
│   ├── data/                 # DataLoader and datasets
│   ├── train/                # Training framework
│   │   ├── module.py         # TrainModule
│   │   ├── trainer.py        # Trainer
│   │   ├── callbacks/        # Callback implementations
│   │   ├── loggers/          # Logger integrations
│   │   └── schedulers/       # LR schedulers
│   ├── models/               # Pre-trained models
│   ├── functional/           # High-level task functions
│   ├── hub/                  # Model registry
│   ├── streaming/            # Real-time audio I/O
│   └── types/                # Result types
├── tests/                    # Test suite
├── examples/                 # Example scripts
├── docs/                     # Documentation
└── pyproject.toml            # Project configuration
```

## Adding New Features

### Adding a New Primitive

1. Create a new module in `mlx_audio/primitives/`:

```python
# mlx_audio/primitives/my_feature.py
"""My new audio feature."""

import mlx.core as mx

def my_feature(
    audio: mx.array,
    param: float = 1.0,
) -> mx.array:
    """Compute my feature from audio.

    Args:
        audio: Input audio array.
        param: Feature parameter.

    Returns:
        Computed feature array.
    """
    # Implementation
    return result
```

2. Export from `mlx_audio/primitives/__init__.py`:

```python
from mlx_audio.primitives.my_feature import my_feature

__all__ = [
    # ... existing exports
    "my_feature",
]
```

3. Add parity tests against a reference implementation (e.g., librosa):

```python
# tests/primitives/test_my_feature.py
import pytest
import numpy as np
import librosa
from mlx_audio.primitives import my_feature

@pytest.mark.parity
def test_my_feature_parity():
    audio = np.random.randn(16000).astype(np.float32)

    result = my_feature(mx.array(audio))
    expected = librosa.some_function(audio)

    np.testing.assert_allclose(
        np.array(result),
        expected,
        rtol=1e-4,
        atol=1e-6,
    )
```

### Adding a New Model

1. Create model directory in `mlx_audio/models/`:

```
mlx_audio/models/my_model/
├── __init__.py
├── model.py        # Model implementation
├── config.py       # Configuration dataclass
└── convert.py      # PyTorch -> MLX weight conversion
```

2. Register with the model registry:

```python
# In mlx_audio/hub/registry.py _register_builtins()

self.register(
    ModelSpec(
        name="my-model",
        task=TaskType.TRANSCRIPTION,  # or appropriate task
        model_class="mlx_audio.models.my_model.MyModel",
        default_repo="mlx-community/my-model",
        supported_repos=[
            "mlx-community/my-model",
            "original/my-model",
        ],
        capabilities=["batched", "streaming"],
    ),
    is_task_default=False,  # Set True if default for task
)
```

### Adding a New Callback

1. Create callback in `mlx_audio/train/callbacks/`:

```python
# mlx_audio/train/callbacks/my_callback.py
from mlx_audio.train.callbacks.base import Callback

class MyCallback(Callback):
    """My custom callback."""

    def __init__(self, param: float = 1.0):
        super().__init__()
        self.param = param

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        """Called after each training batch."""
        if batch_idx % 100 == 0:
            # Do something
            pass
```

2. Export from `mlx_audio/train/callbacks/__init__.py`.

## Pull Request Guidelines

### Before Submitting

1. **Run the test suite**: `pytest tests/`
2. **Check code style**: `ruff check mlx_audio && ruff format --check mlx_audio`
3. **Check types**: `mypy mlx_audio`
4. **Update documentation** if adding new features

### PR Description

Include:
- **Summary**: What changes are made and why
- **Test plan**: How to verify the changes work
- **Breaking changes**: Note any API changes

### Review Process

1. All PRs require at least one review
2. CI must pass (tests, linting, type checking)
3. New features need documentation and tests

## Building Documentation

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build documentation
cd python
mkdocs build

# Serve locally for preview
mkdocs serve
# Open http://127.0.0.1:8000
```

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/zkeown/mlx-audio/issues)
- **Discussions**: [GitHub Discussions](https://github.com/zkeown/mlx-audio/discussions)

Thank you for contributing!
