"""Weight conversion utilities for MusicGen.

Converts HuggingFace MusicGen weights to MLX safetensors format.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_audio.models.base.weight_converter import WeightConverter


class MusicGenConverter(WeightConverter):
    """Converter for MusicGen PyTorch weights to MLX format.

    Handles:
    - Skipping text encoder weights (T5 used separately)
    - Skipping BatchNorm running statistics
    - Key mapping for decoder layers, embeddings, and LM heads

    Example:
        >>> converter = MusicGenConverter()
        >>> converter.convert("musicgen-medium", "musicgen/model.safetensors")
    """

    model_name = "musicgen"

    SKIP_PATTERNS = [
        r"^text_encoder\.",  # T5 used separately
        r"^audio_encoder\.",  # EnCodec used separately
        r"embed_positions\.weights",  # Positional embeddings (built dynamically)
        r"num_batches_tracked",
        r"running_mean",
        r"running_var",
    ]

    KEY_MAPPINGS = [
        # Decoder layers
        (r"^decoder\.model\.decoder\.layers\.(\d+)\.", r"decoder.layers.\1."),
        # Self-attention (already correct)
        (r"\.self_attn\.", ".self_attn."),
        # Cross-attention
        (r"\.encoder_attn\.", ".cross_attn."),
        # Layer norms
        (r"\.self_attn_layer_norm\.", ".self_attn_layer_norm."),
        (r"\.encoder_attn_layer_norm\.", ".cross_attn_layer_norm."),
        (r"\.final_layer_norm\.", ".final_layer_norm."),
        # Projections (already correct)
        (r"\.k_proj\.", ".k_proj."),
        (r"\.v_proj\.", ".v_proj."),
        (r"\.q_proj\.", ".q_proj."),
        (r"\.out_proj\.", ".out_proj."),
        # FFN
        (r"\.fc1\.", ".fc1."),
        (r"\.fc2\.", ".fc2."),
        # Embed tokens (per codebook)
        (r"^decoder\.model\.decoder\.embed_tokens\.(\d+)\.",
         r"embeddings.embeddings.\1."),
        # Final layer norm
        (r"^decoder\.model\.decoder\.layer_norm\.", "decoder.layer_norm."),
        # LM heads (per codebook)
        (r"^decoder\.lm_heads\.(\d+)\.", r"lm_head.linears.\1."),
        # Text projection
        (r"^enc_to_dec_proj\.", "text_projection."),
    ]

    def map_key(self, pt_key: str) -> str | None:
        """Map HuggingFace MusicGen key to MLX key."""
        for pattern in self.SKIP_PATTERNS:
            if re.search(pattern, pt_key):
                return None

        mlx_key = pt_key
        for pattern, replacement in self.KEY_MAPPINGS:
            mlx_key = re.sub(pattern, replacement, mlx_key)

        return mlx_key

    def transform_weight(self, key: str, np_array: np.ndarray) -> np.ndarray:
        """Transform weight array for MLX format.

        Note: Linear layers in PyTorch are [out, in], MLX also uses [out, in].
        No transpose needed for standard Linear layers.
        """
        return np_array


# Module-level converter instance
_converter = MusicGenConverter()


def convert_musicgen_weights(
    pytorch_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, mx.array]:
    """Convert MusicGen PyTorch weights to MLX format.

    Args:
        pytorch_path: Path to HuggingFace model directory
        output_path: Output path for .safetensors file
        config: Optional model configuration to save alongside weights

    Returns:
        Dictionary of MLX arrays
    """
    return _converter.convert(pytorch_path, output_path, config)


def download_and_convert(
    model_name: str = "musicgen-medium",
    output_dir: str | Path | None = None,
) -> Path:
    """Download from HuggingFace and convert to MLX format.

    Args:
        model_name: Model name (musicgen-small, musicgen-medium, etc.)
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
        "musicgen-small": "facebook/musicgen-small",
        "musicgen-medium": "facebook/musicgen-medium",
        "musicgen-large": "facebook/musicgen-large",
        "musicgen-melody": "facebook/musicgen-melody",
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
        from mlx_audio.exceptions import WeightConversionError

        print(f"Downloading {model_name} from HuggingFace...")
        hf_dir = Path(snapshot_download(model_repos[model_name]))

        # Find the checkpoint file - prefer safetensors, fallback to .bin
        checkpoint_file = hf_dir / "model.safetensors"
        if not checkpoint_file.exists():
            checkpoint_file = hf_dir / "pytorch_model.bin"
        if not checkpoint_file.exists():
            raise WeightConversionError(f"No checkpoint found in {hf_dir}")

        # Determine config based on model
        configs = {
            "musicgen-small": {
                "num_codebooks": 4,
                "codebook_size": 2048,
                "hidden_size": 1024,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "intermediate_size": 4096,
                "sample_rate": 32000,
                "frame_rate": 50,
            },
            "musicgen-medium": {
                "num_codebooks": 4,
                "codebook_size": 2048,
                "hidden_size": 1536,
                "num_hidden_layers": 48,
                "num_attention_heads": 24,
                "intermediate_size": 6144,
                "sample_rate": 32000,
                "frame_rate": 50,
            },
            "musicgen-large": {
                "num_codebooks": 4,
                "codebook_size": 2048,
                "hidden_size": 2048,
                "num_hidden_layers": 48,
                "num_attention_heads": 32,
                "intermediate_size": 8192,
                "sample_rate": 32000,
                "frame_rate": 50,
            },
            "musicgen-melody": {
                "num_codebooks": 4,
                "codebook_size": 2048,
                "hidden_size": 1536,
                "num_hidden_layers": 48,
                "num_attention_heads": 24,
                "intermediate_size": 6144,
                "sample_rate": 32000,
                "frame_rate": 50,
            },
        }

        convert_musicgen_weights(checkpoint_file, mlx_path, config=configs[model_name])

    return output_dir


def main():
    """Command-line entry point for weight conversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert MusicGen HuggingFace weights to MLX format"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to HuggingFace model directory",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path for MLX weights (.safetensors)",
    )
    parser.add_argument(
        "-m", "--model",
        choices=[
            "musicgen-small", "musicgen-medium",
            "musicgen-large", "musicgen-melody",
        ],
        help="Download and convert a pretrained model",
    )

    args = parser.parse_args()

    if args.model:
        output_dir = download_and_convert(args.model)
        print(f"\nModel saved to: {output_dir}")
    elif args.input:
        if args.output is None:
            output = Path(args.input) / "model.safetensors"
        else:
            output = Path(args.output)
        convert_musicgen_weights(args.input, output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
