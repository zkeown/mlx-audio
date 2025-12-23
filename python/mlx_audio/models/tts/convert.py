"""Weight conversion utilities for Parler-TTS.

Converts HuggingFace Parler-TTS weights to MLX safetensors format.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


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

    print(f"Loading Parler-TTS weights from {pytorch_path}...")

    # Try to load from HuggingFace format
    if pytorch_path.is_dir():
        # Check for safetensors first
        weights_file = pytorch_path / "model.safetensors"
        if not weights_file.exists():
            weights_file = pytorch_path / "pytorch_model.bin"

        if weights_file.suffix == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(str(weights_file))
        else:
            state_dict = torch.load(
                weights_file, map_location="cpu", weights_only=True
            )
    else:
        state_dict = torch.load(pytorch_path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

    # Convert weights
    print(f"Converting {len(state_dict)} parameters...")
    mlx_weights = {}
    skipped = []

    for pt_key, pt_tensor in state_dict.items():
        # Map PyTorch key to MLX key
        mlx_key = _map_parler_key(pt_key)

        if mlx_key is None:
            skipped.append(pt_key)
            continue

        # Convert to numpy
        if hasattr(pt_tensor, "numpy"):
            np_array = pt_tensor.detach().cpu().numpy()
        else:
            np_array = np.array(pt_tensor)

        # Transform for MLX format
        np_array = _transform_parler_weight(pt_key, np_array)

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


def _map_parler_key(pt_key: str) -> str | None:
    """Map HuggingFace Parler-TTS key to MLX key."""
    # Skip encoder weights (we use T5 separately)
    if pt_key.startswith("text_encoder.") or pt_key.startswith("t5_encoder."):
        return None

    # Skip unused weights
    skip_patterns = [
        r"num_batches_tracked",
        r"running_mean",
        r"running_var",
        r"\.position_ids$",
    ]
    for pattern in skip_patterns:
        if re.search(pattern, pt_key):
            return None

    mlx_key = pt_key

    # Map decoder layers
    # Parler: decoder.layers.{i}.* -> decoder.layers.{i}.*
    mlx_key = re.sub(
        r"^decoder\.layers\.(\d+)\.", r"decoder.layers.\1.", mlx_key
    )

    # Map self-attention
    mlx_key = re.sub(r"\.self_attn\.", ".self_attn.", mlx_key)
    mlx_key = re.sub(r"\.encoder_attn\.", ".encoder_attn.", mlx_key)

    # Map layer norms (RMSNorm in Parler)
    mlx_key = re.sub(
        r"\.self_attn_layer_norm\.", ".self_attn_layer_norm.", mlx_key
    )
    mlx_key = re.sub(
        r"\.encoder_attn_layer_norm\.", ".encoder_attn_layer_norm.", mlx_key
    )
    mlx_key = re.sub(r"\.final_layer_norm\.", ".final_layer_norm.", mlx_key)
    mlx_key = re.sub(r"\.input_layernorm\.", ".self_attn_layer_norm.", mlx_key)
    mlx_key = re.sub(
        r"\.post_attention_layernorm\.", ".encoder_attn_layer_norm.", mlx_key
    )

    # Map projections
    mlx_key = re.sub(r"\.k_proj\.", ".k_proj.", mlx_key)
    mlx_key = re.sub(r"\.v_proj\.", ".v_proj.", mlx_key)
    mlx_key = re.sub(r"\.q_proj\.", ".q_proj.", mlx_key)
    mlx_key = re.sub(r"\.o_proj\.", ".out_proj.", mlx_key)
    mlx_key = re.sub(r"\.out_proj\.", ".out_proj.", mlx_key)

    # Map FFN (SwiGLU: gate_proj, up_proj, down_proj -> fc1, fc2, fc3)
    mlx_key = re.sub(r"\.mlp\.gate_proj\.", ".fc1.", mlx_key)
    mlx_key = re.sub(r"\.mlp\.up_proj\.", ".fc2.", mlx_key)
    mlx_key = re.sub(r"\.mlp\.down_proj\.", ".fc3.", mlx_key)

    # Legacy FFN mapping
    mlx_key = re.sub(r"\.fc1\.", ".fc1.", mlx_key)
    mlx_key = re.sub(r"\.fc2\.", ".fc2.", mlx_key)

    # Map embed tokens (per codebook)
    mlx_key = re.sub(
        r"^decoder\.embed_tokens\.(\d+)\.",
        r"embeddings.embeddings.\1.",
        mlx_key,
    )
    mlx_key = re.sub(
        r"^model\.decoder\.embed_tokens\.(\d+)\.",
        r"embeddings.embeddings.\1.",
        mlx_key,
    )

    # Map final layer norm
    mlx_key = re.sub(
        r"^decoder\.layer_norm\.",
        "decoder.layer_norm.",
        mlx_key,
    )
    mlx_key = re.sub(
        r"^model\.decoder\.norm\.",
        "decoder.layer_norm.",
        mlx_key,
    )

    # Map LM heads (per codebook)
    mlx_key = re.sub(
        r"^lm_heads\.(\d+)\.",
        r"lm_head.linears.\1.",
        mlx_key,
    )

    # Map text/description projections
    mlx_key = re.sub(
        r"^enc_to_dec_proj\.",
        "text_projection.",
        mlx_key,
    )
    mlx_key = re.sub(
        r"^prompt_encoder_proj\.",
        "text_projection.",
        mlx_key,
    )
    mlx_key = re.sub(
        r"^description_encoder_proj\.",
        "description_projection.",
        mlx_key,
    )

    return mlx_key


def _transform_parler_weight(key: str, np_array: np.ndarray) -> np.ndarray:
    """Transform weight array for MLX format."""
    # MLX Linear stores weights as [out, in], same as PyTorch
    # No transformation needed for most weights

    return np_array


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
        raise ImportError(
            "huggingface_hub is required for downloading. "
            "Install with: pip install huggingface-hub"
        )

    model_repos = {
        "parler-tts-mini": "parler-tts/parler-tts-mini-v1",
        "parler-tts-large": "parler-tts/parler-tts-large-v1",
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
