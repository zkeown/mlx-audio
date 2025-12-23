"""Weight conversion utilities for HTDemucs.

Converts PyTorch HTDemucs weights to MLX safetensors format.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


def convert_htdemucs_weights(
    pytorch_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, mx.array]:
    """Convert HTDemucs PyTorch weights to MLX format.

    Args:
        pytorch_path: Path to PyTorch .th or .pth checkpoint
        output_path: Output path for .safetensors file
        config: Optional model configuration to save alongside weights

    Returns:
        Dictionary of MLX arrays

    Example:
        >>> convert_htdemucs_weights(
        ...     "htdemucs_ft.th",
        ...     "htdemucs_ft/model.safetensors"
        ... )
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
    checkpoint = torch.load(pytorch_path, map_location="cpu", weights_only=False)

    # Handle nested checkpoint formats
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state" in checkpoint:
            state_dict = checkpoint["state"]
        else:
            # Assume it's already a state dict
            state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict()

    # Convert weights
    print(f"Converting {len(state_dict)} parameters...")
    mlx_weights = {}
    skipped = []

    for pt_key, pt_tensor in state_dict.items():
        # Map PyTorch key to MLX key
        mlx_key = _map_key(pt_key)

        if mlx_key is None:
            skipped.append(pt_key)
            continue

        # Convert to numpy and transform for MLX format
        np_array = pt_tensor.detach().cpu().numpy()
        np_array = _transform_weight(pt_key, np_array)

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


def _map_key(pt_key: str) -> str | None:
    """Map PyTorch key to MLX key.

    Since our MLX model now matches PyTorch structure exactly,
    we only need to skip BatchNorm running stats (not used in eval).
    """
    # Skip BatchNorm running stats (not needed for inference)
    skip_patterns = [
        r"num_batches_tracked",
        r"running_mean",
        r"running_var",
    ]
    for pattern in skip_patterns:
        if re.search(pattern, pt_key):
            return None

    # No renaming needed - model structure matches PyTorch
    return pt_key


def _transform_weight(key: str, np_array: np.ndarray) -> np.ndarray:
    """Transform weight array for MLX format.

    Handles:
    - Linear layers: PyTorch [out, in] -> MLX [in, out] (transpose)
    - Conv2d: PyTorch [out, in, H, W] -> MLX [out, H, W, in]
    - ConvTranspose2d: PyTorch [in, out, H, W] -> MLX [out, H, W, in]
    - Conv1d: PyTorch [out, in, K] -> MLX [out, K, in]
    - ConvTranspose1d: PyTorch [in, out, K] -> MLX [out, K, in]
    """
    shape = np_array.shape

    # 2D weights: attention in_proj_weight only
    # Note: Linear weights are [out, in] in both PyTorch and MLX - no transpose
    # But in_proj_weight needs transpose for our implementation
    if len(shape) == 2:
        if re.search(r"\.in_proj_weight$", key):
            # in_proj_weight: PyTorch [3*D, D] -> MLX [D, 3*D]
            return np_array.T

    # 4D weights: Conv2d and ConvTranspose2d
    if len(shape) == 4:
        # Check if it's a conv weight (ends with .weight and has 4 dims)
        if key.endswith('.weight'):
            # Check for ConvTranspose2d (conv_tr in decoder)
            if 'conv_tr.weight' in key:
                # ConvTranspose2d: PyTorch [in, out, H, W] -> MLX [out, H, W, in]
                np_array = np.transpose(np_array, (1, 2, 3, 0))
            else:
                # Conv2d: PyTorch [out, in, H, W] -> MLX [out, H, W, in]
                np_array = np.transpose(np_array, (0, 2, 3, 1))
        return np_array

    # 3D weights: Conv1d and ConvTranspose1d
    if len(shape) == 3:
        if key.endswith('.weight'):
            # Check for ConvTranspose1d (conv_tr in time decoder)
            if 'conv_tr.weight' in key:
                # ConvTranspose1d: PyTorch [in, out, K] -> MLX [out, K, in]
                np_array = np.transpose(np_array, (1, 2, 0))
            else:
                # Conv1d: PyTorch [out, in, K] -> MLX [out, K, in]
                np_array = np.transpose(np_array, (0, 2, 1))
        return np_array

    return np_array


def convert_from_demucs_package(
    model_name: str = "htdemucs_ft",
    output_dir: str | Path | None = None,
    model_index: int = 0,
) -> Path:
    """Convert weights using the installed demucs package.

    This is the recommended method as it handles downloading and caching
    automatically via the demucs package.

    Args:
        model_name: Model name (htdemucs, htdemucs_ft, htdemucs_6s)
        output_dir: Output directory (default: ~/.cache/mlx_audio/models)
        model_index: Which model to use from the ensemble (0-3 for htdemucs_ft)

    Returns:
        Path to converted model directory
    """
    try:
        from demucs.pretrained import get_model
    except ImportError:
        raise ImportError(
            "demucs package is required for this conversion method. "
            "Install with: pip install demucs"
        )

    if output_dir is None:
        output_dir = Path.home() / ".cache" / "mlx_audio" / "models"
    output_dir = Path(output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    mlx_path = output_dir / "model.safetensors"

    if not mlx_path.exists():
        print(f"Loading {model_name} via demucs package...")
        bag = get_model(model_name)

        # Get the specific model from the ensemble
        if hasattr(bag, 'models'):
            if model_index >= len(bag.models):
                raise ValueError(
                    f"model_index {model_index} out of range. "
                    f"Ensemble has {len(bag.models)} models."
                )
            model = bag.models[model_index]
            print(f"Using model {model_index} from ensemble of {len(bag.models)}")
        else:
            model = bag

        state_dict = model.state_dict()

        print(f"Converting {len(state_dict)} parameters...")
        mlx_weights = {}
        skipped = []

        for pt_key, pt_tensor in state_dict.items():
            mlx_key = _map_key(pt_key)
            if mlx_key is None:
                skipped.append(pt_key)
                continue

            np_array = pt_tensor.detach().cpu().numpy()
            np_array = _transform_weight(pt_key, np_array)

            mlx_weights[mlx_key] = mx.array(np_array)

        if skipped:
            print(f"Skipped {len(skipped)} parameters (not mapped)")

        # Save weights
        print(f"Saving to {mlx_path}...")
        mx.save_safetensors(str(mlx_path), mlx_weights)

        # Save config
        sources = ["drums", "bass", "other", "vocals"]
        if model_name == "htdemucs_6s":
            sources = ["drums", "bass", "other", "vocals", "guitar", "piano"]

        config = {
            "sources": sources,
            "audio_channels": 2,
            "samplerate": 44100,
            "segment": 6.0,
            "channels": 48,
            "growth": 2.0,
            "depth": 4,
            "nfft": 4096,
            "hop_length": 1024,
            "t_depth": 5,
            "t_heads": 8,
        }

        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved config to {config_path}")

        print(f"Conversion complete! {len(mlx_weights)} parameters saved.")

    return output_dir


def _download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """Download a file with proper headers and progress reporting."""
    import urllib.request

    # Use headers that mimic a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
    }

    request = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(request) as response:
        total_size = response.getheader("Content-Length")
        if total_size:
            total_size = int(total_size)
            print(f"Downloading {total_size / (1024*1024):.1f} MB...")

        with open(dest, "wb") as f:
            downloaded = 0
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    pct = (downloaded / total_size) * 100
                    print(f"\rProgress: {pct:.1f}%", end="", flush=True)
            print()  # newline after progress


def download_and_convert(
    model_name: str = "htdemucs_ft",
    output_dir: str | Path | None = None,
) -> Path:
    """Download PyTorch weights and convert to MLX format.

    Args:
        model_name: Model name (htdemucs, htdemucs_ft, htdemucs_6s)
        output_dir: Output directory (default: ~/.cache/mlx_audio/models)

    Returns:
        Path to converted model directory
    """
    # Model URLs from Facebook's Demucs release
    model_urls = {
        "htdemucs": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/htdemucs-955717e8.th",
        "htdemucs_ft": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/htdemucs_ft-f0ac756c.th",
        "htdemucs_6s": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/htdemucs_6s-cad6e3a7.th",
    }

    if model_name not in model_urls:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(model_urls.keys())}"
        )

    if output_dir is None:
        output_dir = Path.home() / ".cache" / "mlx_audio" / "models"
    output_dir = Path(output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download PyTorch weights
    pytorch_path = output_dir / f"{model_name}.th"
    if not pytorch_path.exists():
        print(f"Downloading {model_name} weights...")
        _download_file(model_urls[model_name], pytorch_path)

    # Convert to MLX
    mlx_path = output_dir / "model.safetensors"
    if not mlx_path.exists():
        # Determine config based on model name
        if model_name == "htdemucs_6s":
            config = {
                "sources": ["drums", "bass", "other", "vocals", "guitar", "piano"],
                "audio_channels": 2,
                "samplerate": 44100,
                "segment": 6.0,
                "channels": 48,
                "growth": 2.0,
                "depth": 4,
                "nfft": 4096,
                "hop_length": 1024,
                "t_depth": 5,
                "t_heads": 8,
            }
        else:
            config = {
                "sources": ["drums", "bass", "other", "vocals"],
                "audio_channels": 2,
                "samplerate": 44100,
                "segment": 6.0,
                "channels": 48,
                "growth": 2.0,
                "depth": 4,
                "nfft": 4096,
                "hop_length": 1024,
                "t_depth": 5,
                "t_heads": 8,
            }

        convert_htdemucs_weights(pytorch_path, mlx_path, config=config)

    return output_dir


def convert_bag_from_demucs_package(
    model_name: str = "htdemucs_ft",
    output_dir: str | Path | None = None,
) -> Path:
    """Convert all models from a BagOfModels ensemble.

    This converts all 4 specialized models from htdemucs_ft to MLX format,
    enabling the full ensemble for improved separation quality.

    Args:
        model_name: Model name (htdemucs_ft has 4 models in ensemble)
        output_dir: Output directory (default: ~/.cache/mlx_audio/models)

    Returns:
        Path to converted bag directory
    """
    try:
        from demucs.pretrained import get_model
    except ImportError:
        raise ImportError(
            "demucs package is required for this conversion method. "
            "Install with: pip install demucs"
        )

    if output_dir is None:
        output_dir = Path.home() / ".cache" / "mlx_audio" / "models"
    output_dir = Path(output_dir) / f"{model_name}_bag"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_name} via demucs package...")
    bag = get_model(model_name)

    if not hasattr(bag, 'models'):
        raise ValueError(f"{model_name} is not a BagOfModels ensemble")

    num_models = len(bag.models)
    print(f"Found {num_models} models in ensemble")

    # Get weight matrix from PyTorch bag
    weights = bag.weights
    print(f"Weight matrix: {weights}")

    # Convert each model
    for i in range(num_models):
        model_dir = output_dir / f"model_{i}"
        mlx_path = model_dir / "model.safetensors"

        if mlx_path.exists():
            print(f"Model {i} already converted, skipping...")
            continue

        model_dir.mkdir(parents=True, exist_ok=True)
        model = bag.models[i]
        state_dict = model.state_dict()

        print(f"Converting model {i} ({len(state_dict)} parameters)...")
        mlx_weights = {}
        skipped = []

        for pt_key, pt_tensor in state_dict.items():
            mlx_key = _map_key(pt_key)
            if mlx_key is None:
                skipped.append(pt_key)
                continue

            np_array = pt_tensor.detach().cpu().numpy()
            np_array = _transform_weight(pt_key, np_array)
            mlx_weights[mlx_key] = mx.array(np_array)

        if skipped:
            print(f"  Skipped {len(skipped)} parameters (not mapped)")

        # Save weights
        mx.save_safetensors(str(mlx_path), mlx_weights)

        # Save config
        sources = ["drums", "bass", "other", "vocals"]
        if model_name == "htdemucs_6s":
            sources = ["drums", "bass", "other", "vocals", "guitar", "piano"]

        config = {
            "sources": sources,
            "audio_channels": 2,
            "samplerate": 44100,
            "segment": 6.0,
            "channels": 48,
            "growth": 2.0,
            "depth": 4,
            "nfft": 4096,
            "hop_length": 1024,
            "t_depth": 5,
            "t_heads": 8,
        }

        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"  Saved model {i} to {model_dir}")

    # Save weight matrix
    weights_path = output_dir / "weights.npy"
    np.save(weights_path, np.array(weights))
    print(f"Saved weight matrix to {weights_path}")

    print(f"\nBag conversion complete! {num_models} models saved to {output_dir}")
    return output_dir


def main():
    """Command-line entry point for weight conversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert HTDemucs PyTorch weights to MLX format"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to PyTorch checkpoint (.th or .pth)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path for MLX weights (.safetensors)",
    )
    parser.add_argument(
        "-m", "--model",
        choices=["htdemucs", "htdemucs_ft", "htdemucs_6s"],
        help="Download and convert a pretrained model",
    )
    parser.add_argument(
        "--bag",
        action="store_true",
        help="Convert full BagOfModels ensemble (for htdemucs_ft)",
    )

    args = parser.parse_args()

    if args.model:
        if args.bag:
            # Convert full bag of models
            output_dir = convert_bag_from_demucs_package(args.model)
        else:
            # Download and convert single model
            output_dir = download_and_convert(args.model)
        print(f"\nModel saved to: {output_dir}")
    elif args.input:
        # Convert local checkpoint
        if args.output is None:
            output = Path(args.input).with_suffix(".safetensors")
        else:
            output = Path(args.output)
        convert_htdemucs_weights(args.input, output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
