"""Weight conversion for Silero VAD models.

Converts PyTorch Silero VAD weights to MLX format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

from mlx_audio.models.vad.config import VADConfig
from mlx_audio.utils.conversion import (
    convert_conv1d_weight,
    extract_state_dict,
    prepare_output_path,
    save_mlx_weights,
    torch_to_numpy,
)

if TYPE_CHECKING:
    pass


def convert_silero_vad_weights(
    input_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 16000,
) -> dict[str, mx.array]:
    """Convert Silero VAD weights from PyTorch/ONNX to MLX format.

    Note: This function handles the conversion of pre-trained Silero VAD
    weights. Since Silero VAD is distributed as JIT/ONNX models without
    source code, this provides a mapping for compatible architectures.

    Args:
        input_path: Path to PyTorch checkpoint or ONNX model
        output_path: Path to save MLX weights
        sample_rate: Sample rate (8000 or 16000) for config

    Returns:
        Dictionary of converted MLX weights

    Example:
        >>> weights = convert_silero_vad_weights(
        ...     "silero_vad.pt",
        ...     "mlx_vad/model.safetensors",
        ... )
    """
    from mlx_audio.utils.dependencies import require_torch

    torch = require_torch()

    input_path = Path(input_path)
    output_path = prepare_output_path(output_path)

    # Load checkpoint
    if input_path.suffix == ".onnx":
        weights = _convert_from_onnx(input_path)
    else:
        checkpoint = torch.load(str(input_path), map_location="cpu", weights_only=False)
        state_dict = extract_state_dict(checkpoint)
        weights = _convert_state_dict(state_dict)

    # Save weights
    save_mlx_weights(weights, output_path)

    # Save config
    config = VADConfig.silero_vad_16k() if sample_rate == 16000 else VADConfig.silero_vad_8k()
    config_path = output_path.parent / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    return weights


def _convert_from_onnx(onnx_path: Path) -> dict[str, mx.array]:
    """Convert ONNX model weights to MLX format.

    Args:
        onnx_path: Path to ONNX model

    Returns:
        Dictionary of MLX weights
    """
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        raise ImportError(
            "onnx is required for ONNX conversion. "
            "Install with: pip install onnx"
        )

    model = onnx.load(str(onnx_path))
    onnx_weights = {
        init.name: numpy_helper.to_array(init)
        for init in model.graph.initializer
    }

    return _map_onnx_weights(onnx_weights)


def _convert_state_dict(state_dict: dict) -> dict[str, mx.array]:
    """Convert PyTorch state dict to MLX weights.

    Args:
        state_dict: PyTorch state dictionary

    Returns:
        Dictionary of MLX weights
    """
    weights = {}

    for name, tensor in state_dict.items():
        np_array = torch_to_numpy(tensor)

        # Convert layer names and handle weight format conversions
        mlx_name, np_array = _convert_weight_name_and_format(name, np_array)

        if mlx_name is not None:
            weights[mlx_name] = mx.array(np_array)

    return weights


def _convert_weight_name_and_format(
    name: str,
    np_array: np.ndarray,
) -> tuple[str | None, np.ndarray]:
    """Convert weight name from PyTorch to MLX format.

    Args:
        name: PyTorch weight name
        np_array: Weight array

    Returns:
        Tuple of (MLX weight name, converted array) or (None, array) to skip
    """
    # Handle Conv1d weights
    if "conv" in name.lower() and "weight" in name.lower() and len(np_array.shape) == 3:
        np_array = convert_conv1d_weight(np_array)

    # Name mapping from typical Silero VAD structure to our architecture
    name_mapping = {
        # Encoder convolutions
        "encoder.conv1.weight": "encoder.conv1.weight",
        "encoder.conv1.bias": "encoder.conv1.bias",
        "encoder.conv2.weight": "encoder.conv2.weight",
        "encoder.conv2.bias": "encoder.conv2.bias",
        "encoder.norm.weight": "encoder.norm.weight",
        "encoder.norm.bias": "encoder.norm.bias",
        # LSTM layers
        "lstm.layers.0.Wii": "lstm.layers.0.Wii",
        "lstm.layers.0.Wif": "lstm.layers.0.Wif",
        "lstm.layers.0.Wig": "lstm.layers.0.Wig",
        "lstm.layers.0.Wio": "lstm.layers.0.Wio",
        "lstm.layers.0.Whi": "lstm.layers.0.Whi",
        "lstm.layers.0.Whf": "lstm.layers.0.Whf",
        "lstm.layers.0.Whg": "lstm.layers.0.Whg",
        "lstm.layers.0.Who": "lstm.layers.0.Who",
        "lstm.layers.0.bias_ih": "lstm.layers.0.bias",
        "lstm.layers.0.bias_hh": "lstm.layers.0.bias_hh",
        "lstm.layers.1.Wii": "lstm.layers.1.Wii",
        "lstm.layers.1.Wif": "lstm.layers.1.Wif",
        "lstm.layers.1.Wig": "lstm.layers.1.Wig",
        "lstm.layers.1.Wio": "lstm.layers.1.Wio",
        "lstm.layers.1.Whi": "lstm.layers.1.Whi",
        "lstm.layers.1.Whf": "lstm.layers.1.Whf",
        "lstm.layers.1.Whg": "lstm.layers.1.Whg",
        "lstm.layers.1.Who": "lstm.layers.1.Who",
        "lstm.layers.1.bias_ih": "lstm.layers.1.bias",
        "lstm.layers.1.bias_hh": "lstm.layers.1.bias_hh",
        # Decoder
        "decoder.fc1.weight": "decoder.fc1.weight",
        "decoder.fc1.bias": "decoder.fc1.bias",
        "decoder.fc2.weight": "decoder.fc2.weight",
        "decoder.fc2.bias": "decoder.fc2.bias",
    }

    # Direct mapping
    if name in name_mapping:
        return name_mapping[name], np_array

    # Try to infer mapping for similar names
    for pt_pattern, mlx_pattern in name_mapping.items():
        if _pattern_match(name, pt_pattern):
            return mlx_pattern, np_array

    # Return original name if no mapping found
    return name, np_array


def _map_onnx_weights(onnx_weights: dict[str, np.ndarray]) -> dict[str, mx.array]:
    """Map ONNX weight names to MLX format.

    ONNX models have different naming conventions than PyTorch.

    Args:
        onnx_weights: Dictionary of ONNX weights

    Returns:
        Dictionary of MLX weights
    """
    weights = {}

    for name, np_array in onnx_weights.items():
        mlx_name, np_array = _convert_weight_name_and_format(name, np_array)
        if mlx_name is not None:
            weights[mlx_name] = mx.array(np_array)

    return weights


def _pattern_match(name: str, pattern: str) -> bool:
    """Simple pattern matching for weight names.

    Args:
        name: Weight name to check
        pattern: Pattern to match against

    Returns:
        True if name matches pattern structure
    """
    # Simple substring matching for now
    return pattern.lower() in name.lower()


def create_random_weights(config: VADConfig) -> dict[str, mx.array]:
    """Create random weights for testing.

    Useful for testing model structure without pre-trained weights.

    Args:
        config: VAD configuration

    Returns:
        Dictionary of random MLX weights
    """
    from mlx_audio.models.vad.model import SileroVAD

    model = SileroVAD(config)

    # Initialize with random weights
    mx.eval(model.parameters())

    return dict(model.parameters())


__all__ = [
    "convert_silero_vad_weights",
    "create_random_weights",
]
