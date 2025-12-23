# Installation

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.11** or higher
- **MLX 0.30.0** or higher

## Install from PyPI

The simplest way to install mlx-audio:

```bash
pip install mlx-audio
```

## Install from Source

For the latest development version:

```bash
git clone https://github.com/zkeown/mlx-audio.git
cd mlx-audio/python
pip install -e .
```

### Development Installation

To install with development dependencies (testing, linting, type checking):

```bash
pip install -e ".[dev]"
```

### Documentation Dependencies

To build the documentation locally:

```bash
pip install -e ".[docs]"
```

## Optional Dependencies

Some features require additional packages:

```bash
# For Weights & Biases logging during training
pip install wandb

# For TensorBoard logging
pip install tensorboard

# For MLflow logging
pip install mlflow
```

## Verify Installation

Test that mlx-audio is properly installed:

```python
import mlx_audio as ma

# Check version
print(ma.__version__)

# Quick functionality test
result = ma.transcribe("path/to/audio.wav")
print(result.text)
```

## C++ Extensions

mlx-audio includes optional C++ extensions for performance-critical operations. These are built automatically during installation when a C++ compiler is available.

To verify C++ extensions are available:

```python
from mlx_audio.primitives import HAS_CPP_EXT
print(f"C++ extensions available: {HAS_CPP_EXT}")
```

If C++ extensions are not available, mlx-audio will gracefully fall back to pure Python implementations.

## Troubleshooting

### MLX Not Found

Ensure you're on Apple Silicon and have MLX installed:

```bash
pip install mlx>=0.30.0
```

### Import Errors

If you encounter import errors, ensure your Python environment is properly configured:

```bash
python -c "import mlx; print(mlx.__version__)"
python -c "import mlx_audio; print('Success!')"
```

### Model Download Issues

Models are automatically downloaded from Hugging Face Hub on first use. If you encounter download issues:

1. Check your internet connection
2. Ensure you have sufficient disk space
3. Try setting the cache directory:

```python
import os
os.environ["HF_HOME"] = "/path/to/cache"
```
