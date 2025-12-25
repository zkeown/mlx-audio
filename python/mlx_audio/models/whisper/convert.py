"""Weight conversion utilities for Whisper.

Converts HuggingFace Whisper weights to MLX safetensors format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_audio.constants import WHISPER_CACHE_DIR
from mlx_audio.utils import (
    convert_conv1d_weight,
    prepare_output_path,
    require_transformers,
    torch_to_numpy,
)


def convert_whisper_weights(
    hf_model_id: str = "openai/whisper-large-v3-turbo",
    output_path: str | Path | None = None,
) -> Path:
    """Convert HuggingFace Whisper weights to MLX format.

    Args:
        hf_model_id: HuggingFace model ID or local path
        output_path: Output directory (default: ~/.cache/mlx_audio/models/whisper)

    Returns:
        Path to converted model directory
    """
    transformers = require_transformers()
    WhisperForConditionalGeneration = transformers.WhisperForConditionalGeneration
    HFWhisperConfig = transformers.WhisperConfig

    # Set output path
    if output_path is None:
        model_name = hf_model_id.split("/")[-1]
        output_path = WHISPER_CACHE_DIR / model_name
    output_path = prepare_output_path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    weights_path = output_path / "model.safetensors"
    config_path = output_path / "config.json"

    if weights_path.exists() and config_path.exists():
        print(f"Model already exists at {output_path}")
        return output_path

    # Load HuggingFace model
    print(f"Loading HuggingFace model: {hf_model_id}")
    hf_model = WhisperForConditionalGeneration.from_pretrained(hf_model_id)
    hf_config = HFWhisperConfig.from_pretrained(hf_model_id)

    # Extract state dict
    state_dict = hf_model.state_dict()
    print(f"Found {len(state_dict)} parameters")

    # Convert weights
    mlx_weights = _convert_encoder(state_dict)
    mlx_weights.update(_convert_decoder(state_dict))

    print(f"Converted {len(mlx_weights)} parameters")

    # Save weights
    print(f"Saving to {weights_path}")
    mx.save_safetensors(str(weights_path), mlx_weights)

    # Extract and save config
    config = _extract_whisper_config(hf_config)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    return output_path


def _convert_encoder(state_dict: dict) -> dict:
    """Convert Whisper encoder weights."""
    weights = {}

    for hf_key, tensor in state_dict.items():
        if not hf_key.startswith("model.encoder."):
            continue

        np_array = torch_to_numpy(tensor)

        # Map key structure
        key = hf_key.replace("model.encoder.", "encoder.")

        # Handle conv layers
        if "conv1." in key or "conv2." in key:
            if ".weight" in key:
                np_array = convert_conv1d_weight(np_array)
            weights[key] = mx.array(np_array)
            continue

        # Handle embed_positions - this is the sinusoidal positional embedding
        if "embed_positions.weight" in key:
            # HF stores as [max_len, dim], we store as [max_len, dim]
            weights["encoder.positional_embedding"] = mx.array(np_array)
            continue

        # Handle transformer layers
        if ".layers." in key:
            key = _map_encoder_layer_key(key)
            if key is not None:
                weights[key] = mx.array(np_array)
            continue

        # Handle layer norm
        if "layer_norm." in key:
            mlx_key = key.replace("layer_norm.", "ln_post.")
            weights[mlx_key] = mx.array(np_array)
            continue

    return weights


def _map_encoder_layer_key(hf_key: str) -> str | None:
    """Map HuggingFace encoder layer key to MLX key."""
    # Pattern: encoder.layers.{i}.self_attn.{q/k/v/out}_proj.{weight/bias}
    # Target:  encoder.blocks.{i}.attn.{query/key/value/out}.{weight/bias}

    key = hf_key

    # Map layer index
    key = key.replace("encoder.layers.", "encoder.blocks.")

    # Map self attention components
    if ".self_attn." in key:
        key = key.replace(".self_attn.", ".attn.")
        key = key.replace(".q_proj.", ".query.")
        key = key.replace(".k_proj.", ".key.")
        key = key.replace(".v_proj.", ".value.")
        key = key.replace(".out_proj.", ".out.")
        return key

    # Map layer norms
    if ".self_attn_layer_norm." in key:
        key = key.replace(".self_attn_layer_norm.", ".attn_ln.")
        return key

    # Map FFN
    if ".fc1." in key:
        key = key.replace(".fc1.", ".mlp.layers.0.")
        return key
    if ".fc2." in key:
        key = key.replace(".fc2.", ".mlp.layers.2.")
        return key

    if ".final_layer_norm." in key:
        key = key.replace(".final_layer_norm.", ".mlp_ln.")
        return key

    return None


def _convert_decoder(state_dict: dict) -> dict:
    """Convert Whisper decoder weights."""
    weights = {}

    for hf_key, tensor in state_dict.items():
        if not hf_key.startswith("model.decoder."):
            continue

        np_array = torch_to_numpy(tensor)

        # Map key structure
        key = hf_key.replace("model.decoder.", "decoder.")

        # Handle embed_tokens
        if "embed_tokens.weight" in key:
            weights["decoder.token_embedding.weight"] = mx.array(np_array)
            continue

        # Handle embed_positions
        if "embed_positions.weight" in key:
            weights["decoder.positional_embedding"] = mx.array(np_array)
            continue

        # Handle transformer layers
        if ".layers." in key:
            key = _map_decoder_layer_key(key)
            if key is not None:
                weights[key] = mx.array(np_array)
            continue

        # Handle final layer norm
        if "layer_norm." in key:
            mlx_key = key.replace("layer_norm.", "ln.")
            weights[mlx_key] = mx.array(np_array)
            continue

    return weights


def _map_decoder_layer_key(hf_key: str) -> str | None:
    """Map HuggingFace decoder layer key to MLX key."""
    key = hf_key

    # Map layer index
    key = key.replace("decoder.layers.", "decoder.blocks.")

    # Map self attention
    if ".self_attn." in key:
        key = key.replace(".self_attn.", ".attn.")
        key = key.replace(".q_proj.", ".query.")
        key = key.replace(".k_proj.", ".key.")
        key = key.replace(".v_proj.", ".value.")
        key = key.replace(".out_proj.", ".out.")
        return key

    if ".self_attn_layer_norm." in key:
        key = key.replace(".self_attn_layer_norm.", ".attn_ln.")
        return key

    # Map cross attention
    if ".encoder_attn." in key:
        key = key.replace(".encoder_attn.", ".cross_attn.")
        key = key.replace(".q_proj.", ".query.")
        key = key.replace(".k_proj.", ".key.")
        key = key.replace(".v_proj.", ".value.")
        key = key.replace(".out_proj.", ".out.")
        return key

    if ".encoder_attn_layer_norm." in key:
        key = key.replace(".encoder_attn_layer_norm.", ".cross_attn_ln.")
        return key

    # Map FFN
    if ".fc1." in key:
        key = key.replace(".fc1.", ".mlp.layers.0.")
        return key
    if ".fc2." in key:
        key = key.replace(".fc2.", ".mlp.layers.2.")
        return key

    if ".final_layer_norm." in key:
        key = key.replace(".final_layer_norm.", ".mlp_ln.")
        return key

    return None


def convert_weights(state_dict: dict) -> dict[str, np.ndarray]:
    """Convert HuggingFace state dict to MLX-compatible numpy arrays.

    This is useful for testing numerical parity between implementations.

    Args:
        state_dict: HuggingFace model state dict

    Returns:
        Dictionary of numpy arrays with MLX-compatible key names
    """
    weights = {}

    # Convert encoder weights
    encoder_weights = _convert_encoder(state_dict)
    for k, v in encoder_weights.items():
        weights[k] = np.array(v)

    # Convert decoder weights
    decoder_weights = _convert_decoder(state_dict)
    for k, v in decoder_weights.items():
        weights[k] = np.array(v)

    return weights


def _extract_whisper_config(hf_config) -> dict[str, Any]:
    """Extract MLX config from HuggingFace config."""
    return {
        "n_mels": hf_config.num_mel_bins,
        "n_audio_ctx": hf_config.max_source_positions,
        "n_audio_state": hf_config.d_model,
        "n_audio_head": hf_config.encoder_attention_heads,
        "n_audio_layer": hf_config.encoder_layers,
        "n_text_ctx": hf_config.max_target_positions,
        "n_text_state": hf_config.d_model,
        "n_text_head": hf_config.decoder_attention_heads,
        "n_text_layer": hf_config.decoder_layers,
        "n_vocab": hf_config.vocab_size,
    }


def download_and_convert(
    model_name: str = "whisper-large-v3-turbo",
    output_dir: str | Path | None = None,
) -> Path:
    """Download and convert a Whisper model.

    Args:
        model_name: Model name (whisper-tiny, whisper-base, etc.)
        output_dir: Output directory

    Returns:
        Path to converted model directory
    """
    # Map short names to HuggingFace IDs
    model_map = {
        "whisper-tiny": "openai/whisper-tiny",
        "whisper-tiny.en": "openai/whisper-tiny.en",
        "whisper-base": "openai/whisper-base",
        "whisper-base.en": "openai/whisper-base.en",
        "whisper-small": "openai/whisper-small",
        "whisper-small.en": "openai/whisper-small.en",
        "whisper-medium": "openai/whisper-medium",
        "whisper-medium.en": "openai/whisper-medium.en",
        "whisper-large": "openai/whisper-large",
        "whisper-large-v2": "openai/whisper-large-v2",
        "whisper-large-v3": "openai/whisper-large-v3",
        "whisper-large-v3-turbo": "openai/whisper-large-v3-turbo",
    }

    hf_model_id = model_map.get(model_name, model_name)
    return convert_whisper_weights(hf_model_id, output_dir)


def main():
    """Command-line entry point for weight conversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert HuggingFace Whisper weights to MLX format"
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="openai/whisper-large-v3-turbo",
        help="HuggingFace model ID or path",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output directory",
    )

    args = parser.parse_args()

    output_path = convert_whisper_weights(args.model, args.output)
    print(f"\nModel saved to: {output_path}")


if __name__ == "__main__":
    main()
