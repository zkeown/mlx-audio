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

from mlx_audio.models.base.weight_converter import WeightConverter


class BanquetConverter(WeightConverter):
    """Converter for Banquet PyTorch weights to MLX format.

    Handles:
    - Skipping optimizer/scheduler/loss states
    - Key mapping for band_split, tf_model, film, mask_estim modules
    - Transposing linear and conv layers

    Example:
        >>> converter = BanquetConverter()
        >>> converter.convert("ev-pre-aug.ckpt", "banquet/model.safetensors")
    """

    model_name = "banquet"

    SKIP_PATTERNS = [
        r"^optimizer\.",
        r"^scheduler\.",
        r"^loss\.",
        r"^ema\.",
        r"num_batches_tracked",
        r"running_mean",
        r"running_var",
    ]

    KEY_MAPPINGS = [
        # Band split
        (r"^band_split\.norm_fc\.(\d+)\.", r"band_split.norm_fc_modules.\1."),
        # TF model (seqband)
        (r"^tf_model\.seqband\.(\d+)\.rnn_forward\.",
         r"tf_model.seqband.\1.rnn."),
        (r"^tf_model\.seqband\.(\d+)\.rnn_backward\.",
         r"tf_model.seqband.\1.rnn_reverse."),
        # FiLM
        (r"^conditioner\.film\.", r"film."),
        (r"^film\.gn\.", r"film.group_norm."),
        # Mask estimation
        (r"^mask_estim\.band_mlps\.(\d+)\.", r"mask_estim.norm_mlp.\1."),
        (r"^mask_estim\.norm_mlp\.(\d+)\.hidden\.",
         r"mask_estim.norm_mlp.\1.hidden_linear."),
        (r"^mask_estim\.norm_mlp\.(\d+)\.output\.",
         r"mask_estim.norm_mlp.\1.output_linear."),
    ]

    def map_key(self, pt_key: str) -> str | None:
        """Map PyTorch Banquet key to MLX key."""
        for pattern in self.SKIP_PATTERNS:
            if re.search(pattern, pt_key):
                return None

        mlx_key = pt_key
        for pattern, replacement in self.KEY_MAPPINGS:
            mlx_key = re.sub(pattern, replacement, mlx_key)

        return mlx_key

    def transform_weight(self, key: str, np_array: np.ndarray) -> np.ndarray:
        """Transform Banquet weight array for MLX format."""
        shape = np_array.shape

        # Linear layer weights (2D)
        if len(shape) == 2 and key.endswith(".weight"):
            # Don't transpose LSTM weights
            if "rnn" in key.lower() or "lstm" in key.lower():
                return np_array
            return np_array.T

        # LSTM weights: keep as is
        if "rnn" in key.lower() or "lstm" in key.lower():
            if "weight_ih" in key or "weight_hh" in key:
                return np_array

        # Conv2d weights (4D)
        if len(shape) == 4 and key.endswith(".weight"):
            return np.transpose(np_array, (0, 2, 3, 1))

        return np_array


class PaSSTConverter(WeightConverter):
    """Converter for PaSST PyTorch weights to MLX format.

    PaSST is the audio encoder used by Banquet for conditioning.

    Handles:
    - Key mapping for patch_embed, blocks, position embeddings
    - Transposing linear and conv layers
    - Skipping classification head (encoder only)

    Example:
        >>> converter = PaSSTConverter()
        >>> converter.convert("passt.pt", "passt/model.safetensors")
    """

    model_name = "passt"

    SKIP_PATTERNS = [
        r"^head\.",
        r"^head_dist\.",
        r"^pre_logits\.",
        r"num_batches_tracked",
    ]

    KEY_MAPPINGS = [
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

    def map_key(self, pt_key: str) -> str | None:
        """Map PyTorch PaSST key to MLX key."""
        for pattern in self.SKIP_PATTERNS:
            if re.search(pattern, pt_key):
                return None

        mlx_key = pt_key
        for pattern, replacement in self.KEY_MAPPINGS:
            mlx_key = re.sub(pattern, replacement, mlx_key)

        return mlx_key

    def transform_weight(self, key: str, np_array: np.ndarray) -> np.ndarray:
        """Transform PaSST weight array for MLX format."""
        shape = np_array.shape

        # Linear layer weights (2D) - need transpose
        if len(shape) == 2 and key.endswith(".weight"):
            return np_array.T

        # Conv2d weights (4D) - patch embedding
        if len(shape) == 4 and key.endswith(".weight"):
            return np.transpose(np_array, (0, 2, 3, 1))

        return np_array


# Module-level converter instances
_banquet_converter = BanquetConverter()
_passt_converter = PaSSTConverter()


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
    return _banquet_converter.convert(pytorch_path, output_path, config)


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
    return _passt_converter.convert(pytorch_path, output_path, config)


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

    zenodo_url = "https://zenodo.org/records/13694558/files/checkpoints.zip"

    print("Downloading Banquet checkpoints from Zenodo...")
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "checkpoints.zip"

        print("This may take a while (~100MB)...")
        urllib.request.urlretrieve(zenodo_url, zip_path)

        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        ckpt_path = Path(tmpdir) / "checkpoints" / f"{checkpoint_name}.ckpt"
        if not ckpt_path.exists():
            for p in Path(tmpdir).rglob(f"*{checkpoint_name}*.ckpt"):
                ckpt_path = p
                break

        if not ckpt_path.exists():
            from mlx_audio.exceptions import WeightConversionError
            raise WeightConversionError(
                f"Could not find {checkpoint_name}.ckpt in downloaded archive"
            )

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
        from mlx_audio.exceptions import WeightConversionError
        raise WeightConversionError(
            "hear21passt package is required. "
            "Install with: pip install hear21passt"
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

    converter = PaSSTConverter()

    print(f"Converting {len(state_dict)} parameters...")
    mlx_weights = {}
    skipped = []

    for pt_key, pt_tensor in state_dict.items():
        mlx_key = converter.map_key("net." + pt_key)

        if mlx_key is None:
            skipped.append(pt_key)
            continue

        np_array = pt_tensor.detach().cpu().numpy()
        np_array = converter.transform_weight(pt_key, np_array)

        mlx_weights[mlx_key] = mx.array(np_array)

    if skipped:
        print(f"Skipped {len(skipped)} parameters")

    print(f"Saving to {mlx_path}...")
    mx.save_safetensors(str(mlx_path), mlx_weights)

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
