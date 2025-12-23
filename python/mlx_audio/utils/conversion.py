"""Weight conversion utilities for PyTorch to MLX.

Provides common functions used across model conversion scripts
for loading, transforming, and saving weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


def prepare_output_path(output_path: str | Path) -> Path:
    """Ensure output directory exists.

    Args:
        output_path: Path to output file or directory

    Returns:
        Path object with parent directories created
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    """Extract state dict from various checkpoint formats.

    Handles different checkpoint formats:
    - Direct state dict
    - Dict with "state_dict", "model", or "state" key
    - Module with state_dict() method

    Args:
        checkpoint: Loaded checkpoint in various formats

    Returns:
        State dictionary mapping parameter names to tensors
    """
    if isinstance(checkpoint, dict):
        # Try common keys for nested state dicts
        for key in ("state_dict", "model", "state"):
            if key in checkpoint:
                return checkpoint[key]
        # Assume it's already a state dict
        return checkpoint

    # Try calling state_dict() method
    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()

    raise ValueError(
        f"Cannot extract state dict from checkpoint of type {type(checkpoint)}"
    )


def convert_conv1d_weight(np_array: np.ndarray) -> np.ndarray:
    """Convert Conv1d weight from PyTorch to MLX format.

    PyTorch Conv1d: [out_channels, in_channels, kernel_size]
    MLX Conv1d:     [out_channels, kernel_size, in_channels]

    Args:
        np_array: Weight array in PyTorch format

    Returns:
        Weight array in MLX format
    """
    if len(np_array.shape) != 3:
        raise ValueError(f"Expected 3D array, got shape {np_array.shape}")
    return np.transpose(np_array, (0, 2, 1))


def convert_conv_transpose1d_weight(np_array: np.ndarray) -> np.ndarray:
    """Convert ConvTranspose1d weight from PyTorch to MLX format.

    PyTorch ConvTranspose1d: [in_channels, out_channels, kernel_size]
    MLX ConvTranspose1d:     [out_channels, kernel_size, in_channels]

    Args:
        np_array: Weight array in PyTorch format

    Returns:
        Weight array in MLX format
    """
    if len(np_array.shape) != 3:
        raise ValueError(f"Expected 3D array, got shape {np_array.shape}")
    return np.transpose(np_array, (1, 2, 0))


def convert_weight(
    np_array: np.ndarray,
    weight_type: str | None = None,
) -> mx.array:
    """Convert numpy weight to MLX with appropriate transformations.

    Args:
        np_array: Weight array in numpy format
        weight_type: Type of weight for format conversion:
            - "conv1d": Apply Conv1d transpose
            - "conv_transpose1d": Apply ConvTranspose1d transpose
            - None: No transformation

    Returns:
        MLX array with appropriate format
    """
    if weight_type == "conv1d" and len(np_array.shape) == 3:
        np_array = convert_conv1d_weight(np_array)
    elif weight_type == "conv_transpose1d" and len(np_array.shape) == 3:
        np_array = convert_conv_transpose1d_weight(np_array)

    return mx.array(np_array)


def torch_to_numpy(tensor) -> np.ndarray:
    """Convert PyTorch tensor to numpy array.

    Handles CUDA tensors by moving to CPU first.

    Args:
        tensor: PyTorch tensor

    Returns:
        Numpy array
    """
    return tensor.detach().cpu().numpy()


def save_mlx_weights(
    weights: dict[str, mx.array],
    output_path: str | Path,
) -> None:
    """Save MLX weights to safetensors format.

    Args:
        weights: Dictionary of MLX arrays
        output_path: Path to save safetensors file
    """
    output_path = prepare_output_path(output_path)
    mx.save_safetensors(str(output_path), weights)


def load_pytorch_checkpoint(
    path: str | Path,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load a PyTorch checkpoint file.

    Requires torch to be installed.

    Args:
        path: Path to checkpoint file
        map_location: Device to load tensors to (default: "cpu")

    Returns:
        Loaded checkpoint (state dict or full checkpoint)
    """
    from mlx_audio.utils.dependencies import require_torch

    torch = require_torch()

    path = Path(path)

    # Handle safetensors format
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file

            return load_file(str(path))
        except ImportError:
            raise ImportError(
                "safetensors is required to load .safetensors files. "
                "Install with: pip install safetensors"
            )

    # Load regular PyTorch checkpoint
    return torch.load(str(path), map_location=map_location, weights_only=False)


__all__ = [
    "prepare_output_path",
    "extract_state_dict",
    "convert_conv1d_weight",
    "convert_conv_transpose1d_weight",
    "convert_weight",
    "torch_to_numpy",
    "save_mlx_weights",
    "load_pytorch_checkpoint",
]
