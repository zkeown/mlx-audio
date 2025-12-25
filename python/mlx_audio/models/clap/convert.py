"""Weight conversion utilities for CLAP.

Converts HuggingFace CLAP weights to MLX safetensors format.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


def convert_clap_weights(
    hf_model_id: str = "laion/clap-htsat-fused",
    output_path: str | Path | None = None,
    include_text_encoder: bool = True,
) -> Path:
    """Convert HuggingFace CLAP weights to MLX format.

    Args:
        hf_model_id: HuggingFace model ID or local path
        output_path: Output directory (default: ~/.cache/mlx_audio/models/clap)
        include_text_encoder: Whether to include text encoder weights

    Returns:
        Path to converted model directory
    """
    try:
        from transformers import ClapConfig as HFClapConfig
        from transformers import ClapModel
    except ImportError:
        from mlx_audio.exceptions import WeightConversionError
        raise WeightConversionError(
            "transformers is required for CLAP conversion. "
            "Install with: pip install transformers"
        )

    # Set output path
    if output_path is None:
        model_name = hf_model_id.split("/")[-1]
        output_path = Path.home() / ".cache" / "mlx_audio" / "models" / "clap" / model_name
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    weights_path = output_path / "model.safetensors"
    config_path = output_path / "config.json"

    if weights_path.exists() and config_path.exists():
        print(f"Model already exists at {output_path}")
        return output_path

    # Load HuggingFace model
    print(f"Loading HuggingFace model: {hf_model_id}")
    hf_model = ClapModel.from_pretrained(hf_model_id)
    hf_config = HFClapConfig.from_pretrained(hf_model_id)

    # Extract state dict
    state_dict = hf_model.state_dict()
    print(f"Found {len(state_dict)} parameters")

    # Convert weights
    mlx_weights = _convert_audio_encoder(state_dict)

    if include_text_encoder:
        mlx_weights.update(_convert_text_encoder(state_dict))

    # Convert projection layers
    mlx_weights.update(_convert_projections(state_dict))

    print(f"Converted {len(mlx_weights)} parameters")

    # Save weights
    print(f"Saving to {weights_path}")
    mx.save_safetensors(str(weights_path), mlx_weights)

    # Extract and save config
    config = _extract_clap_config(hf_config)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    return output_path


def _convert_audio_encoder(state_dict: dict) -> dict:
    """Convert HTSAT audio encoder weights."""
    weights = {}

    # Collect QKV weights by block for merging
    qkv_buffer = {}

    for hf_key, tensor in state_dict.items():
        if not hf_key.startswith("audio_model.audio_encoder."):
            continue

        np_array = tensor.detach().cpu().numpy()

        # Skip certain buffers
        if "relative_position_index" in hf_key:
            continue

        # Handle batch norm on frequency dimension
        if ".batch_norm." in hf_key:
            # Skip num_batches_tracked (not used in MLX)
            if "num_batches_tracked" in hf_key:
                continue
            key = hf_key.replace("audio_model.audio_encoder.", "audio_encoder.")
            weights[key] = mx.array(np_array)
            continue

        # Strip prefix
        key = hf_key.replace("audio_model.audio_encoder.", "audio_encoder.")

        # Handle QKV - need to merge separate q/k/v into combined qkv
        if ".attention.self." in key:
            # Parse the block location
            match = re.match(r"audio_encoder\.layers\.(\d+)\.blocks\.(\d+)\.attention\.self\.(query|key|value)\.(weight|bias)", key)
            if match:
                layer_idx, block_idx, qkv_type, param_type = match.groups()
                block_key = f"audio_encoder.layers.{layer_idx}.blocks.{block_idx}"

                if block_key not in qkv_buffer:
                    qkv_buffer[block_key] = {}
                if param_type not in qkv_buffer[block_key]:
                    qkv_buffer[block_key][param_type] = {}

                qkv_buffer[block_key][param_type][qkv_type] = np_array
                continue

            # Handle relative position bias table
            if "relative_position_bias_table" in key:
                mlx_key = re.sub(r"\.attention\.self\.", ".attn.", key)
                weights[mlx_key] = mx.array(np_array)
                continue

        # Handle attention output projection
        # Note: MLX uses same [out, in] weight shape as PyTorch, no transpose
        if ".attention.output.dense." in key:
            mlx_key = key.replace(".attention.output.dense.", ".attn.proj.")
            weights[mlx_key] = mx.array(np_array)
            continue

        # Handle LayerNorm
        if ".layernorm_before." in key:
            mlx_key = key.replace(".layernorm_before.", ".norm1.")
            weights[mlx_key] = mx.array(np_array)
            continue
        if ".layernorm_after." in key:
            mlx_key = key.replace(".layernorm_after.", ".norm2.")
            weights[mlx_key] = mx.array(np_array)
            continue

        # Handle MLP intermediate/output
        # Note: MLX uses same [out, in] weight shape as PyTorch, no transpose
        if ".intermediate.dense." in key:
            mlx_key = key.replace(".intermediate.dense.", ".mlp.fc1.")
            weights[mlx_key] = mx.array(np_array)
            continue
        if ".output.dense." in key:
            mlx_key = key.replace(".output.dense.", ".mlp.fc2.")
            weights[mlx_key] = mx.array(np_array)
            continue

        # Handle downsample
        # Note: MLX uses same [out, in] weight shape as PyTorch, no transpose
        if ".downsample." in key:
            weights[key] = mx.array(np_array)
            continue

        # Handle patch embed
        if "patch_embed.proj." in key:
            if key.endswith(".weight"):
                # Conv2d: [out, in, H, W] -> [out, H, W, in]
                np_array = np.transpose(np_array, (0, 2, 3, 1))
            weights[key] = mx.array(np_array)
            continue
        if "patch_embed.norm." in key:
            weights[key] = mx.array(np_array)
            continue

        # Handle fusion components (mel_conv2d and fusion_model/AFF block)
        if "patch_embed.mel_conv2d." in key:
            if key.endswith(".weight"):
                # Conv2d: [out, in, H, W] -> [out, H, W, in]
                np_array = np.transpose(np_array, (0, 2, 3, 1))
            weights[key] = mx.array(np_array)
            continue

        # Handle fusion_model (AFFBlock) - Conv2d/BatchNorm layers
        if "patch_embed.fusion_model." in key:
            # Skip num_batches_tracked (not used in MLX)
            if "num_batches_tracked" in key:
                continue

            # HF global_att has AdaptiveAvgPool at index 0, shift indices
            # HF: [0:AvgPool, 1:Conv, 2:BN, 3:ReLU, 4:Conv, 5:BN]
            # MLX: [0:Conv, 1:BN, 2:ReLU, 3:Conv, 4:BN]
            if ".global_att." in key:
                index_map = {1: 0, 2: 1, 4: 3, 5: 4}
                match = re.search(r"\.global_att\.(\d+)\.", key)
                if match:
                    hf_idx = int(match.group(1))
                    if hf_idx in index_map:
                        mlx_idx = index_map[hf_idx]
                        key = re.sub(
                            r"\.global_att\.\d+\.",
                            f".global_att.{mlx_idx}.",
                            key
                        )

            if ".weight" in key and "running_" not in key:
                # Check if it's a Conv2d weight (4D)
                if len(np_array.shape) == 4:
                    # Conv2d: [out, in, H, W] -> [out, H, W, in]
                    np_array = np.transpose(np_array, (0, 2, 3, 1))
            weights[key] = mx.array(np_array)
            continue

        # Handle norm layer
        if "norm." in key and "layernorm" not in key and "patch_embed" not in key:
            weights[key] = mx.array(np_array)
            continue

        # Note: No fc layer in HuggingFace HTSAT - removed

    # Now merge QKV weights
    for block_key, params in qkv_buffer.items():
        for param_type, qkv_parts in params.items():
            if len(qkv_parts) == 3:
                # Stack q, k, v
                q = qkv_parts["query"]
                k = qkv_parts["key"]
                v = qkv_parts["value"]

                if param_type == "weight":
                    # Concatenate: each is [dim, in], result is [3*dim, in]
                    # MLX Linear expects [out, in] = [3*dim, in]
                    merged = np.concatenate([q, k, v], axis=0)
                else:
                    # Bias: just concatenate
                    merged = np.concatenate([q, k, v], axis=0)

                mlx_key = f"{block_key}.attn.qkv.{param_type}"
                weights[mlx_key] = mx.array(merged)

    return weights


def _convert_text_encoder(state_dict: dict) -> dict:
    """Convert RoBERTa text encoder weights."""
    weights = {}

    for hf_key, tensor in state_dict.items():
        if not hf_key.startswith("text_model."):
            continue

        np_array = tensor.detach().cpu().numpy()

        # Skip position IDs and token_type_ids (buffers, not parameters)
        if "position_ids" in hf_key:
            continue
        if "token_type_ids" in hf_key:
            continue

        # Map text model keys
        key = hf_key.replace("text_model.", "text_encoder.")

        # Handle embeddings
        if ".embeddings." in key:
            weights[key] = mx.array(np_array)
            continue

        # Handle encoder layers
        if ".encoder.layer." in key:
            key = key.replace(".encoder.layer.", ".encoder.layers.")

            # Self attention - no transpose, same [out, in] format
            if ".attention.self." in key:
                mlx_key = key.replace(".attention.self.", ".attention.self_attn.")
                weights[mlx_key] = mx.array(np_array)
                continue

            # Attention output - no transpose, same [out, in] format
            if ".attention.output.dense." in key:
                mlx_key = key.replace(".attention.output.dense.", ".attention.output.")
                weights[mlx_key] = mx.array(np_array)
                continue
            if ".attention.output.LayerNorm." in key:
                mlx_key = key.replace(".attention.output.LayerNorm.", ".attention.layer_norm.")
                weights[mlx_key] = mx.array(np_array)
                continue

            # Intermediate/output - no transpose, same [out, in] format
            if ".intermediate.dense." in key:
                weights[key] = mx.array(np_array)
                continue
            if ".output.dense." in key and ".attention." not in key:
                weights[key] = mx.array(np_array)
                continue
            if ".output.LayerNorm." in key and ".attention." not in key:
                mlx_key = key.replace(".output.LayerNorm.", ".output.layer_norm.")
                weights[mlx_key] = mx.array(np_array)
                continue

        # Pooler - no transpose, same [out, in] format
        if ".pooler." in key:
            weights[key] = mx.array(np_array)
            continue

    return weights


def _convert_projections(state_dict: dict) -> dict:
    """Convert projection layer weights."""
    weights = {}

    for hf_key, tensor in state_dict.items():
        np_array = tensor.detach().cpu().numpy()

        # Audio projection - no transpose, same [out, in] format
        if hf_key.startswith("audio_projection."):
            weights[hf_key] = mx.array(np_array)
            continue

        # Text projection (into text_encoder.projection) - no transpose
        if hf_key.startswith("text_projection."):
            mlx_key = hf_key.replace("text_projection.", "text_encoder.projection.")
            weights[mlx_key] = mx.array(np_array)
            continue

        # Logit scale (use audio scale) - ensure shape is (1,) not ()
        if hf_key == "logit_scale_a":
            weights["logit_scale"] = mx.array([np_array.item()])
            continue

    return weights


def _extract_clap_config(hf_config) -> dict[str, Any]:
    """Extract MLX config from HuggingFace config."""
    audio_config = {
        "sample_rate": 48000,
        "n_mels": 64,
        "n_fft": 1024,
        "hop_length": 480,
        "window_length": 1024,
        "patch_size": getattr(hf_config.audio_config, "patch_size", 4),
        "patch_stride": getattr(hf_config.audio_config, "patch_stride", 4),
        "embed_dim": getattr(hf_config.audio_config, "patch_embeds_hidden_size", 96),
        "depths": list(getattr(hf_config.audio_config, "depths", [2, 2, 6, 2])),
        "num_heads": list(getattr(hf_config.audio_config, "num_attention_heads", [4, 8, 16, 32])),
        "window_size": getattr(hf_config.audio_config, "window_size", 8),
        "mlp_ratio": getattr(hf_config.audio_config, "mlp_ratio", 4.0),
        "hidden_size": getattr(hf_config.audio_config, "hidden_size", 768),
        "enable_fusion": getattr(hf_config.audio_config, "enable_fusion", True),
        "fusion_type": getattr(hf_config.audio_config, "fusion_type", "aff_2d"),
    }

    text_config = {
        "vocab_size": getattr(hf_config.text_config, "vocab_size", 50265),
        "hidden_size": getattr(hf_config.text_config, "hidden_size", 768),
        "num_hidden_layers": getattr(hf_config.text_config, "num_hidden_layers", 12),
        "num_attention_heads": getattr(hf_config.text_config, "num_attention_heads", 12),
        "intermediate_size": getattr(hf_config.text_config, "intermediate_size", 3072),
        "max_position_embeddings": getattr(hf_config.text_config, "max_position_embeddings", 514),
    }

    return {
        "audio": audio_config,
        "text": text_config,
        "projection_dim": getattr(hf_config, "projection_dim", 512),
        "logit_scale_init": getattr(hf_config, "logit_scale_init_value", 2.6592),
    }


def download_and_convert(
    model_name: str = "clap-htsat-fused",
    output_dir: str | Path | None = None,
) -> Path:
    """Download and convert a CLAP model.

    Args:
        model_name: Model name (clap-htsat-fused, clap-htsat-unfused, etc.)
        output_dir: Output directory

    Returns:
        Path to converted model directory
    """
    # Map short names to HuggingFace IDs
    model_map = {
        "clap-htsat-fused": "laion/clap-htsat-fused",
        "clap-htsat-unfused": "laion/clap-htsat-unfused",
        "larger_clap_music": "laion/larger_clap_music",
        "larger_clap_general": "laion/larger_clap_general",
        "larger_clap_music_and_speech": "laion/larger_clap_music_and_speech",
    }

    hf_model_id = model_map.get(model_name, model_name)
    return convert_clap_weights(hf_model_id, output_dir)


def main():
    """Command-line entry point for weight conversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert HuggingFace CLAP weights to MLX format"
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="laion/clap-htsat-fused",
        help="HuggingFace model ID or path",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output directory",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Exclude text encoder weights",
    )

    args = parser.parse_args()

    output_path = convert_clap_weights(
        args.model,
        args.output,
        include_text_encoder=not args.no_text,
    )
    print(f"\nModel saved to: {output_path}")


if __name__ == "__main__":
    main()
