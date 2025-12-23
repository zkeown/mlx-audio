"""Weight conversion utilities for EnCodec.

Converts PyTorch EnCodec weights to MLX safetensors format.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


def convert_encodec_weights(
    pytorch_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, mx.array]:
    """Convert EnCodec PyTorch weights to MLX format.

    Args:
        pytorch_path: Path to PyTorch checkpoint or HuggingFace model directory
        output_path: Output path for .safetensors file
        config: Optional model configuration to save alongside weights

    Returns:
        Dictionary of MLX arrays
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for weight conversion. "
            "Install with: pip install torch"
        )

    pytorch_path = Path(pytorch_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load PyTorch checkpoint
    print(f"Loading PyTorch checkpoint from {pytorch_path}...")

    if pytorch_path.is_dir():
        # HuggingFace format
        weights_file = pytorch_path / "pytorch_model.bin"
        if not weights_file.exists():
            weights_file = pytorch_path / "model.safetensors"
        state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)
    else:
        checkpoint = torch.load(pytorch_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        else:
            state_dict = checkpoint.state_dict()

    # Convert weights
    print(f"Converting {len(state_dict)} parameters...")
    mlx_weights = {}
    skipped = []

    for pt_key, pt_tensor in state_dict.items():
        # Map PyTorch key to MLX key
        mlx_key = _map_encodec_key(pt_key)

        if mlx_key is None:
            skipped.append(pt_key)
            continue

        # Convert to numpy and transform for MLX format
        np_array = pt_tensor.detach().cpu().numpy()
        np_array = _transform_encodec_weight(pt_key, np_array)

        # Convert to MLX array
        mlx_weights[mlx_key] = mx.array(np_array)

    if skipped:
        print(f"Skipped {len(skipped)} parameters (not mapped)")

    # Save as safetensors
    print(f"Saving to {output_path}...")
    mx.save_safetensors(str(output_path), mlx_weights)

    # Save config if provided
    if config is not None:
        config_path = output_path.with_name("config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved config to {config_path}")

    print(f"Conversion complete! {len(mlx_weights)} parameters saved.")
    return mlx_weights


def _map_encodec_key(pt_key: str) -> str | None:
    """Map PyTorch EnCodec key to MLX key."""
    # Skip BatchNorm running stats
    skip_patterns = [
        r"num_batches_tracked",
        r"running_mean",
        r"running_var",
    ]
    for pattern in skip_patterns:
        if re.search(pattern, pt_key):
            return None

    # Map encoder/decoder structure
    key_mappings = [
        # Encoder mappings
        (r"^encoder\.", "encoder."),
        # Decoder mappings
        (r"^decoder\.", "decoder."),
        # Quantizer mappings
        (r"^quantizer\.vq\.", "quantizer.layers."),
        (r"^quantizer\.codebook\.", "quantizer.layers."),
    ]

    mlx_key = pt_key
    for pattern, replacement in key_mappings:
        mlx_key = re.sub(pattern, replacement, mlx_key)

    return mlx_key


def _transform_encodec_weight(key: str, np_array: np.ndarray) -> np.ndarray:
    """Transform weight array for MLX format."""
    shape = np_array.shape

    # Conv1d: PyTorch [out, in, K] -> MLX [out, K, in]
    if len(shape) == 3 and key.endswith('.weight'):
        if 'conv' in key.lower():
            np_array = np.transpose(np_array, (0, 2, 1))

    # ConvTranspose1d: PyTorch [in, out, K] -> MLX [out, K, in]
    if len(shape) == 3 and 'transpose' in key.lower():
        np_array = np.transpose(np_array, (1, 2, 0))

    return np_array


def download_and_convert(
    model_name: str = "encodec_32khz",
    output_dir: str | Path | None = None,
) -> Path:
    """Download from HuggingFace and convert to MLX format.

    Args:
        model_name: Model name (encodec_24khz, encodec_32khz, encodec_48khz)
        output_dir: Output directory (default: ~/.cache/mlx_audio/models)

    Returns:
        Path to converted model directory
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for downloading. "
            "Install with: pip install huggingface-hub"
        )

    model_repos = {
        "encodec_24khz": "facebook/encodec_24khz",
        "encodec_32khz": "facebook/encodec_32khz",
        "encodec_48khz": "facebook/encodec_48khz",
    }

    if model_name not in model_repos:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(model_repos.keys())}"
        )

    if output_dir is None:
        output_dir = Path.home() / ".cache" / "mlx_audio" / "models"
    output_dir = Path(output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    mlx_path = output_dir / "model.safetensors"

    if not mlx_path.exists():
        print(f"Downloading {model_name} from HuggingFace...")
        hf_dir = snapshot_download(model_repos[model_name])

        # Determine config based on model
        configs = {
            "encodec_24khz": {
                "sample_rate": 24000,
                "channels": 1,
                "num_codebooks": 8,
                "codebook_size": 1024,
                "codebook_dim": 128,
                "ratios": [8, 5, 4, 2],
            },
            "encodec_32khz": {
                "sample_rate": 32000,
                "channels": 1,
                "num_codebooks": 4,
                "codebook_size": 2048,
                "codebook_dim": 128,
                "ratios": [8, 5, 4, 4],
            },
            "encodec_48khz": {
                "sample_rate": 48000,
                "channels": 2,
                "num_codebooks": 8,
                "codebook_size": 1024,
                "codebook_dim": 128,
                "ratios": [8, 5, 4, 2],
            },
        }

        convert_encodec_weights(hf_dir, mlx_path, config=configs[model_name])

    return output_dir


def main():
    """Command-line entry point for weight conversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert EnCodec PyTorch weights to MLX format"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to PyTorch checkpoint or HuggingFace model directory",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path for MLX weights (.safetensors)",
    )
    parser.add_argument(
        "-m", "--model",
        choices=["encodec_24khz", "encodec_32khz", "encodec_48khz"],
        help="Download and convert a pretrained model",
    )

    args = parser.parse_args()

    if args.model:
        output_dir = download_and_convert(args.model)
        print(f"\nModel saved to: {output_dir}")
    elif args.input:
        if args.output is None:
            output = Path(args.input).with_suffix(".safetensors")
        else:
            output = Path(args.output)
        convert_encodec_weights(args.input, output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
