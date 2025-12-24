"""Convert PyTorch DrumTranscriber weights to MLX format.

Usage:
    python -m mlx_audio.models.drums.convert /path/to/checkpoint.pt /path/to/output/
"""

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np


def convert_conv2d_weights(torch_weight: np.ndarray) -> np.ndarray:
    """Convert Conv2d weights from PyTorch to MLX format.

    PyTorch: (out_channels, in_channels, height, width)
    MLX: (out_channels, height, width, in_channels)
    """
    return np.transpose(torch_weight, (0, 2, 3, 1))


def convert_linear_weights(torch_weight: np.ndarray) -> np.ndarray:
    """Convert Linear weights from PyTorch to MLX format.

    PyTorch: (out_features, in_features)
    MLX: (out_features, in_features) - same, but may need transpose
    """
    # MLX uses the same format for Linear weights
    return torch_weight


def convert_state_dict(torch_state: dict) -> dict:
    """Convert PyTorch state dict to MLX format.

    Args:
        torch_state: PyTorch model state dict

    Returns:
        MLX-compatible weight dict
    """
    mlx_weights = {}

    for key, value in torch_state.items():
        # Convert tensor to numpy
        if hasattr(value, "numpy"):
            value = value.numpy()
        elif hasattr(value, "cpu"):
            value = value.cpu().numpy()

        new_key = key

        # Handle Conv2d weights
        if ".conv.weight" in key or "conv1.conv.weight" in key or "conv2.conv.weight" in key:
            value = convert_conv2d_weights(value)

        # Handle BatchNorm -> BatchNorm
        # PyTorch uses running_mean, running_var
        # MLX BatchNorm expects these as well
        if "running_mean" in key:
            new_key = key.replace("running_mean", "running_mean")
        elif "running_var" in key:
            new_key = key.replace("running_var", "running_var")

        # Handle projection convolutions
        if ".proj_conv.weight" in key and value.ndim == 4:
            value = convert_conv2d_weights(value)

        # Handle QKV linear weights - PyTorch stores as (3*embed, embed)
        # MLX expects same format
        if ".qkv.weight" in key:
            # Linear weights are same format
            pass

        mlx_weights[new_key] = value

    return mlx_weights


def load_pytorch_checkpoint(checkpoint_path: Path) -> dict:
    """Load PyTorch checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file

    Returns:
        Model state dict
    """
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    else:
        return checkpoint


def save_mlx_weights(weights: dict, output_dir: Path) -> None:
    """Save MLX weights.

    Args:
        weights: MLX weight dict
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to MLX arrays and save
    mlx_weights = {k: mx.array(v) for k, v in weights.items()}

    # Save as safetensors
    weights_path = output_dir / "model.safetensors"
    mx.save_safetensors(str(weights_path), mlx_weights)

    print(f"Saved weights to {weights_path}")
    print(f"Total parameters: {sum(v.size for v in mlx_weights.values()):,}")


def convert_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
) -> None:
    """Convert PyTorch checkpoint to MLX format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_dir: Output directory for MLX weights
    """
    print(f"Loading PyTorch checkpoint: {checkpoint_path}")
    torch_state = load_pytorch_checkpoint(checkpoint_path)

    print(f"Converting {len(torch_state)} tensors...")
    mlx_weights = convert_state_dict(torch_state)

    print(f"Saving MLX weights to: {output_dir}")
    save_mlx_weights(mlx_weights, output_dir)

    print("Conversion complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch DrumTranscriber weights to MLX format"
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to PyTorch checkpoint (.pt file)",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output directory for MLX weights",
    )

    args = parser.parse_args()

    convert_checkpoint(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
