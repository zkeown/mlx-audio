"""Weight conversion utilities for Parler-TTS.

Converts HuggingFace Parler-TTS weights to MLX safetensors format.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_audio.models.base.weight_converter import WeightConverter


class ParlerTTSConverter(WeightConverter):
    """Converter for Parler-TTS PyTorch weights to MLX format.

    Handles:
    - Skipping text encoder weights (T5 used separately)
    - Skipping BatchNorm running statistics
    - Key mapping for decoder layers, embeddings, projections, and LM heads

    Example:
        >>> converter = ParlerTTSConverter()
        >>> converter.convert("parler-tts-mini", "parler/model.safetensors")
    """

    model_name = "parler-tts"

    SKIP_PATTERNS = [
        r"^text_encoder\.",
        r"^t5_encoder\.",
        r"num_batches_tracked",
        r"running_mean",
        r"running_var",
        r"\.position_ids$",
    ]

    KEY_MAPPINGS = [
        # Decoder layers
        (r"^decoder\.layers\.(\d+)\.", r"decoder.layers.\1."),
        # Self-attention
        (r"\.self_attn\.", ".self_attn."),
        (r"\.encoder_attn\.", ".encoder_attn."),
        # Layer norms (RMSNorm in Parler)
        (r"\.self_attn_layer_norm\.", ".self_attn_layer_norm."),
        (r"\.encoder_attn_layer_norm\.", ".encoder_attn_layer_norm."),
        (r"\.final_layer_norm\.", ".final_layer_norm."),
        (r"\.input_layernorm\.", ".self_attn_layer_norm."),
        (r"\.post_attention_layernorm\.", ".encoder_attn_layer_norm."),
        # Projections
        (r"\.k_proj\.", ".k_proj."),
        (r"\.v_proj\.", ".v_proj."),
        (r"\.q_proj\.", ".q_proj."),
        (r"\.o_proj\.", ".out_proj."),
        (r"\.out_proj\.", ".out_proj."),
        # FFN (SwiGLU: gate_proj, up_proj, down_proj -> fc1, fc2, fc3)
        (r"\.mlp\.gate_proj\.", ".fc1."),
        (r"\.mlp\.up_proj\.", ".fc2."),
        (r"\.mlp\.down_proj\.", ".fc3."),
        # Legacy FFN mapping
        (r"\.fc1\.", ".fc1."),
        (r"\.fc2\.", ".fc2."),
        # Embed tokens (per codebook)
        (r"^decoder\.embed_tokens\.(\d+)\.", r"embeddings.embeddings.\1."),
        (r"^model\.decoder\.embed_tokens\.(\d+)\.",
         r"embeddings.embeddings.\1."),
        # Final layer norm
        (r"^decoder\.layer_norm\.", "decoder.layer_norm."),
        (r"^model\.decoder\.norm\.", "decoder.layer_norm."),
        # LM heads (per codebook)
        (r"^lm_heads\.(\d+)\.", r"lm_head.linears.\1."),
        # Text/description projections
        (r"^enc_to_dec_proj\.", "text_projection."),
        (r"^prompt_encoder_proj\.", "text_projection."),
        (r"^description_encoder_proj\.", "description_projection."),
    ]

    def map_key(self, pt_key: str) -> str | None:
        """Map HuggingFace Parler-TTS key to MLX key."""
        for pattern in self.SKIP_PATTERNS:
            if re.search(pattern, pt_key):
                return None

        mlx_key = pt_key
        for pattern, replacement in self.KEY_MAPPINGS:
            mlx_key = re.sub(pattern, replacement, mlx_key)

        return mlx_key

    def transform_weight(self, key: str, np_array: np.ndarray) -> np.ndarray:
        """Transform weight array for MLX format.

        Note: MLX Linear stores weights as [out, in], same as PyTorch.
        No transformation needed for most weights.
        """
        return np_array


# Module-level converter instance
_converter = ParlerTTSConverter()


def convert_parler_weights(
    pytorch_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, mx.array]:
    """Convert Parler-TTS PyTorch weights to MLX format.

    Args:
        pytorch_path: Path to HuggingFace model directory
        output_path: Output path for .safetensors file
        config: Optional model configuration to save alongside weights

    Returns:
        Dictionary of MLX arrays
    """
    return _converter.convert(pytorch_path, output_path, config)


def download_and_convert(
    model_name: str = "parler-tts-mini",
    output_dir: str | Path | None = None,
) -> Path:
    """Download from HuggingFace and convert to MLX format.

    Args:
        model_name: Model name (parler-tts-mini, parler-tts-large)
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
        "parler-tts-mini": "parler-tts/parler-tts-mini-v1",
        "parler-tts-large": "parler-tts/parler-tts-large-v1",
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
            "parler-tts-mini": {
                "num_codebooks": 9,
                "codebook_size": 1024,
                "hidden_size": 1024,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "num_key_value_heads": 16,
                "intermediate_size": 4096,
                "sample_rate": 24000,
                "frame_rate": 75,
                "text_encoder_name": "google/flan-t5-large",
                "text_hidden_size": 1024,
                "description_encoder_name": "google/flan-t5-large",
                "description_hidden_size": 1024,
            },
            "parler-tts-large": {
                "num_codebooks": 9,
                "codebook_size": 1024,
                "hidden_size": 1536,
                "num_hidden_layers": 36,
                "num_attention_heads": 24,
                "num_key_value_heads": 24,
                "intermediate_size": 6144,
                "sample_rate": 24000,
                "frame_rate": 75,
                "text_encoder_name": "google/flan-t5-large",
                "text_hidden_size": 1024,
                "description_encoder_name": "google/flan-t5-large",
                "description_hidden_size": 1024,
            },
        }

        convert_parler_weights(hf_dir, mlx_path, config=configs[model_name])

    return output_dir


def main():
    """Command-line entry point for weight conversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Parler-TTS HuggingFace weights to MLX format"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to HuggingFace model directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output path for MLX weights (.safetensors)",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=["parler-tts-mini", "parler-tts-large"],
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
        convert_parler_weights(args.input, output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
