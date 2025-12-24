# Contributing to mlx-audio

Thank you for your interest in contributing! This guide will help you get started.

## Quick Start

```bash
# Clone and install
git clone https://github.com/zkeown/mlx-audio.git
cd mlx-audio/python
pip install -e ".[dev]"

# Run tests
pytest tests/

# Check code style
ruff check mlx_audio
ruff format --check mlx_audio
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- Git

## Code Style

- **Line length**: 100 characters
- **Type hints**: Required for all public functions
- **Docstrings**: Google style
- **Formatting**: Run `ruff format mlx_audio`

## Before Submitting a PR

1. Run tests: `pytest tests/`
2. Check linting: `ruff check mlx_audio`
3. Check formatting: `ruff format --check mlx_audio`
4. Update docs if needed

## Full Guide

For detailed instructions on:

- Adding new primitives
- Adding new models
- Adding callbacks
- Project structure

See the [full contributing guide](../python/docs/contributing.md).

## Getting Help

- [GitHub Issues](https://github.com/zkeown/mlx-audio/issues) - Bug reports
- [GitHub Discussions](https://github.com/zkeown/mlx-audio/discussions) - Questions and ideas
