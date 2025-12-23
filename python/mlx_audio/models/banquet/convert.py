"""Weight conversion utilities for Banquet.

Converts PyTorch Banquet and PaSST weights to MLX safetensors format.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


def convert_banquet_weights(
    pytorch_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, mx.array]:
    """Convert Banquet PyTorch weights to MLX format.

    Args:
        pytorch_path: Path to PyTorch checkpoint (.ckpt or .pt)
        output_path: Output path for .safetensors file
        config: Optional model configuration to save alongside weights

    Returns:
        Dictionary of MLX arrays

    Example:
        >>> convert_banquet_weights(
        ...     "ev-pre-aug.ckpt",
        ...     "banquet/model.safetensors"
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
            state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict()

    # Convert weights
    print(f"Converting {len(state_dict)} parameters...")
    mlx_weights = {}
    skipped = []

    for pt_key, pt_tensor in state_dict.items():
        # Map PyTorch key to MLX key
        mlx_key = _map_banquet_key(pt_key)

        if mlx_key is None:
            skipped.append(pt_key)
            continue

        # Convert to numpy and transform for MLX format
        np_array = pt_tensor.detach().cpu().numpy()
        np_array = _transform_banquet_weight(pt_key, np_array)

        # Convert to MLX array
        mlx_weights[mlx_key] = mx.array(np_array)

    if skipped:
        print(f"Skipped {len(skipped)} parameters:")
        for key in skipped[:10]:
            print(f"  - {key}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

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


def convert_passt_weights(
    pytorch_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, mx.array]:
    """Convert PaSST PyTorch weights to MLX format.

    Args:
        pytorch_path: Path to PyTorch checkpoint or state dict
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
    checkpoint = torch.load(pytorch_path, map_location="cpu", weights_only=False)

    # Handle nested checkpoint formats
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict()

    # Convert weights
    print(f"Converting {len(state_dict)} parameters...")
    mlx_weights = {}
    skipped = []

    for pt_key, pt_tensor in state_dict.items():
        # Map PyTorch key to MLX key
        mlx_key = _map_passt_key(pt_key)

        if mlx_key is None:
            skipped.append(pt_key)
            continue

        # Convert to numpy and transform for MLX format
        np_array = pt_tensor.detach().cpu().numpy()
        np_array = _transform_passt_weight(pt_key, np_array)

        # Convert to MLX array
        mlx_weights[mlx_key] = mx.array(np_array)

    if skipped:
        print(f"Skipped {len(skipped)} parameters")

    # Save as safetensors
    print(f"Saving to {output_path}...")
    mx.save_safetensors(str(output_path), mlx_weights)

    # Save config if provided
    if config is not None:
        config_path = output_path.with_name("passt_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved config to {config_path}")

    print(f"Conversion complete! {len(mlx_weights)} parameters saved.")
    return mlx_weights


def _map_banquet_key(pt_key: str) -> str | None:
    """Map PyTorch Banquet key to MLX key.

    Banquet model structure:
        - band_split.norm_fc_modules.{i}.norm/fc
        - tf_model.seqband.{i}.norm/rnn/fc
        - film.gamma.{i}/beta.{i}
        - mask_estim.norm_mlp.{i}.norm/hidden_linear/output_linear
    """
    # Skip optimizer states, loss function, etc.
    skip_patterns = [
        r"^optimizer\.",
        r"^scheduler\.",
        r"^loss\.",
        r"^ema\.",
        r"num_batches_tracked",
        r"running_mean",
        r"running_var",
    ]
    for pattern in skip_patterns:
        if re.search(pattern, pt_key):
            return None

    # Map key prefixes
    key_mappings = [
        # Band split
        (r"^band_split\.norm_fc\.(\d+)\.", r"band_split.norm_fc_modules.\1."),
        # TF model (seqband)
        (r"^tf_model\.seqband\.(\d+)\.rnn_forward\.", r"tf_model.seqband.\1.rnn."),
        (r"^tf_model\.seqband\.(\d+)\.rnn_backward\.", r"tf_model.seqband.\1.rnn_reverse."),
        # FiLM
        (r"^conditioner\.film\.", r"film."),
        (r"^film\.gn\.", r"film.group_norm."),
        # Mask estimation
        (r"^mask_estim\.band_mlps\.(\d+)\.", r"mask_estim.norm_mlp.\1."),
        (r"^mask_estim\.norm_mlp\.(\d+)\.hidden\.", r"mask_estim.norm_mlp.\1.hidden_linear."),
        (r"^mask_estim\.norm_mlp\.(\d+)\.output\.", r"mask_estim.norm_mlp.\1.output_linear."),
    ]

    mlx_key = pt_key
    for pattern, replacement in key_mappings:
        mlx_key = re.sub(pattern, replacement, mlx_key)

    return mlx_key


def _transform_banquet_weight(key: str, np_array: np.ndarray) -> np.ndarray:
    """Transform Banquet weight array for MLX format.

    Handles:
    - Linear layers: PyTorch [out, in] -> MLX [in, out] (transpose)
    - Conv2d: PyTorch [out, in, H, W] -> MLX [out, H, W, in]
    - LSTM weights need special handling
    """
    shape = np_array.shape

    # Linear layer weights (2D)
    if len(shape) == 2 and key.endswith(".weight"):
        # Check if it's part of an LSTM (don't transpose)
        if "rnn" in key.lower() or "lstm" in key.lower():
            return np_array
        # Regular linear: transpose
        return np_array.T

    # LSTM weights: may need transposition depending on MLX convention
    if "rnn" in key.lower() or "lstm" in key.lower():
        if "weight_ih" in key or "weight_hh" in key:
            # LSTM weights: [4*hidden, input/hidden] -> keep as is for MLX
            return np_array

    # Conv2d weights (4D)
    if len(shape) == 4 and key.endswith(".weight"):
        # PyTorch [out, in, H, W] -> MLX [out, H, W, in]
        return np.transpose(np_array, (0, 2, 3, 1))

    return np_array


def _map_passt_key(pt_key: str) -> str | None:
    """Map PyTorch PaSST key to MLX key.

    PaSST model structure:
        - patch_embed.proj (Conv2d)
        - cls_token, dist_token
        - time_pos_embed, freq_pos_embed, pos_embed
        - blocks.{i}.norm1/attn/norm2/mlp
        - norm (final layer norm)
    """
    # Skip classification head (we only need encoder)
    skip_patterns = [
        r"^head\.",
        r"^head_dist\.",
        r"^pre_logits\.",
        r"num_batches_tracked",
    ]
    for pattern in skip_patterns:
        if re.search(pattern, pt_key):
            return None

    # Map hear21passt specific key patterns
    key_mappings = [
        # Patch embedding
        (r"^net\.patch_embed\.proj\.", r"patch_embed.proj."),
        (r"^patch_embed\.proj\.", r"patch_embed.proj."),
        # Position embeddings
        (r"^net\.time_new_pos_embed$", r"time_pos_embed"),
        (r"^net\.freq_new_pos_embed$", r"freq_pos_embed"),
        (r"^net\.new_pos_embed$", r"pos_embed"),
        (r"^time_new_pos_embed$", r"time_pos_embed"),
        (r"^freq_new_pos_embed$", r"freq_pos_embed"),
        (r"^new_pos_embed$", r"pos_embed"),
        # Special tokens
        (r"^net\.cls_token$", r"cls_token"),
        (r"^net\.dist_token$", r"dist_token"),
        # Transformer blocks
        (r"^net\.blocks\.(\d+)\.", r"blocks.\1."),
        # Final norm
        (r"^net\.norm\.", r"norm."),
    ]

    mlx_key = pt_key
    for pattern, replacement in key_mappings:
        mlx_key = re.sub(pattern, replacement, mlx_key)

    return mlx_key


def _transform_passt_weight(key: str, np_array: np.ndarray) -> np.ndarray:
    """Transform PaSST weight array for MLX format."""
    shape = np_array.shape

    # Linear layer weights (2D) - need transpose
    if len(shape) == 2 and key.endswith(".weight"):
        return np_array.T

    # Conv2d weights (4D) - patch embedding
    if len(shape) == 4 and key.endswith(".weight"):
        # PyTorch [out, in, H, W] -> MLX [out, H, W, in]
        return np.transpose(np_array, (0, 2, 3, 1))

    return np_array


def convert_from_zenodo(
    output_dir: str | Path | None = None,
    checkpoint_name: str = "ev-pre-aug",
) -> Path:
    """Download and convert Banquet weights from Zenodo.

    The official Banquet checkpoints are available at:
    https://zenodo.org/records/13694558

    Args:
        output_dir: Output directory (default: ~/.cache/mlx_audio/models/banquet)
        checkpoint_name: Checkpoint variant (ev-pre-aug, etc.)

    Returns:
        Path to converted model directory
    """
    import tempfile
    import urllib.request
    import zipfile

    if output_dir is None:
        output_dir = Path.home() / ".cache" / "mlx_audio" / "models" / "banquet"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mlx_path = output_dir / "model.safetensors"

    if mlx_path.exists():
        print(f"Model already exists at {mlx_path}")
        return output_dir

    # Zenodo download URL for Banquet checkpoints
    zenodo_url = "https://zenodo.org/records/13694558/files/checkpoints.zip"

    print(f"Downloading Banquet checkpoints from Zenodo...")
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "checkpoints.zip"

        # Download
        print("This may take a while (~100MB)...")
        urllib.request.urlretrieve(zenodo_url, zip_path)

        # Extract
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        # Find the checkpoint
        ckpt_path = Path(tmpdir) / "checkpoints" / f"{checkpoint_name}.ckpt"
        if not ckpt_path.exists():
            # Try alternative locations
            for p in Path(tmpdir).rglob(f"*{checkpoint_name}*.ckpt"):
                ckpt_path = p
                break

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Could not find {checkpoint_name}.ckpt in downloaded archive"
            )

        # Convert
        config = {
            "sample_rate": 44100,
            "n_fft": 2048,
            "hop_length": 512,
            "in_channel": 2,
            "n_bands": 64,
            "emb_dim": 128,
            "rnn_dim": 256,
            "n_sqm_modules": 12,
            "mlp_dim": 512,
            "bidirectional": True,
            "rnn_type": "LSTM",
            "complex_mask": True,
            "cond_emb_dim": 768,
            "film_additive": True,
            "film_multiplicative": True,
            "film_depth": 2,
        }

        convert_banquet_weights(ckpt_path, mlx_path, config=config)

    return output_dir


def convert_passt_from_hear21passt(
    output_dir: str | Path | None = None,
) -> Path:
    """Convert PaSST weights from hear21passt package.

    Requires hear21passt package: pip install hear21passt

    Args:
        output_dir: Output directory

    Returns:
        Path to converted model directory
    """
    try:
        from hear21passt.base import get_basic_model
    except ImportError:
        raise ImportError(
            "hear21passt package is required. Install with: pip install hear21passt"
        )

    if output_dir is None:
        output_dir = Path.home() / ".cache" / "mlx_audio" / "models" / "passt"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mlx_path = output_dir / "model.safetensors"

    if mlx_path.exists():
        print(f"Model already exists at {mlx_path}")
        return output_dir

    print("Loading PaSST from hear21passt...")
    model = get_basic_model(mode="embed_only", arch="openmic")
    state_dict = model.net.state_dict()

    print(f"Converting {len(state_dict)} parameters...")
    mlx_weights = {}
    skipped = []

    for pt_key, pt_tensor in state_dict.items():
        import torch
        mlx_key = _map_passt_key("net." + pt_key)

        if mlx_key is None:
            skipped.append(pt_key)
            continue

        np_array = pt_tensor.detach().cpu().numpy()
        np_array = _transform_passt_weight(pt_key, np_array)

        mlx_weights[mlx_key] = mx.array(np_array)

    if skipped:
        print(f"Skipped {len(skipped)} parameters")

    # Save
    print(f"Saving to {mlx_path}...")
    mx.save_safetensors(str(mlx_path), mlx_weights)

    # Save config
    config = {
        "sample_rate": 32000,
        "n_mels": 128,
        "n_fft": 1024,
        "hop_length": 320,
        "win_length": 800,
        "embed_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "mlp_ratio": 4.0,
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Conversion complete! Saved to {output_dir}")
    return output_dir


def main():
    """Command-line entry point for weight conversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Banquet/PaSST PyTorch weights to MLX format"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to PyTorch checkpoint (.ckpt or .pt)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path for MLX weights (.safetensors)",
    )
    parser.add_argument(
        "--banquet",
        action="store_true",
        help="Download and convert official Banquet weights from Zenodo",
    )
    parser.add_argument(
        "--passt",
        action="store_true",
        help="Convert PaSST from hear21passt package",
    )
    parser.add_argument(
        "--checkpoint",
        default="ev-pre-aug",
        help="Banquet checkpoint name (default: ev-pre-aug)",
    )

    args = parser.parse_args()

    if args.banquet:
        output_dir = convert_from_zenodo(
            output_dir=args.output,
            checkpoint_name=args.checkpoint,
        )
        print(f"\nBanquet model saved to: {output_dir}")
    elif args.passt:
        output_dir = convert_passt_from_hear21passt(output_dir=args.output)
        print(f"\nPaSST model saved to: {output_dir}")
    elif args.input:
        if args.output is None:
            output = Path(args.input).with_suffix(".safetensors")
        else:
            output = Path(args.output)
        convert_banquet_weights(args.input, output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
