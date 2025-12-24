"""Weight conversion utilities for EnCodec.

Converts PyTorch EnCodec weights to MLX safetensors format.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_audio.models.base.weight_converter import WeightConverter


class EnCodecConverter(WeightConverter):
    """Converter for EnCodec PyTorch weights to MLX format.

    Handles:
    - Skipping BatchNorm running statistics
    - Transposing conv kernels from PyTorch to MLX format
    - Key mapping for encoder/decoder/quantizer modules

    Example:
        >>> converter = EnCodecConverter()
        >>> converter.convert("encodec.pt", "encodec/model.safetensors")
    """

    model_name = "encodec"

    SKIP_PATTERNS = [
        r"num_batches_tracked",
        r"running_mean",
        r"running_var",
    ]

    KEY_MAPPINGS = [
        # Encoder mappings
        (r"^encoder\.", "encoder."),
        # Decoder mappings
        (r"^decoder\.", "decoder."),
        # Quantizer mappings
        (r"^quantizer\.vq\.", "quantizer.layers."),
        (r"^quantizer\.codebook\.", "quantizer.layers."),
    ]

    def map_key(self, pt_key: str) -> str | None:
        """Map PyTorch EnCodec key to MLX key."""
        for pattern in self.SKIP_PATTERNS:
            if re.search(pattern, pt_key):
                return None

        mlx_key = pt_key
        for pattern, replacement in self.KEY_MAPPINGS:
            mlx_key = re.sub(pattern, replacement, mlx_key)

        return mlx_key

    def transform_weight(self, key: str, np_array: np.ndarray) -> np.ndarray:
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


# Module-level converter instance
_converter = EnCodecConverter()


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
    return _converter.convert(pytorch_path, output_path, config)


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
        from mlx_audio.exceptions import WeightConversionError
        raise WeightConversionError(
            "huggingface_hub is required for downloading. "
            "Install with: pip install huggingface-hub"
        )

    model_repos = {
        "encodec_24khz": "facebook/encodec_24khz",
        "encodec_32khz": "facebook/encodec_32khz",
        "encodec_48khz": "facebook/encodec_48khz",
    }

    if model_name not in model_repos:
        from mlx_audio.exceptions import ConfigurationError
        available = list(model_repos.keys())
        raise ConfigurationError(
            f"Unknown model: {model_name}. Available: {available}"
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
