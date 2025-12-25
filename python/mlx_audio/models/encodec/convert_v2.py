"""Weight conversion for EnCodec from HuggingFace to MLX.

Properly handles:
1. Weight normalization (g, v -> weight)
2. Layer index mapping
3. LSTM weight format differences
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def weight_norm_to_weight(weight_g: np.ndarray, weight_v: np.ndarray) -> np.ndarray:
    """Convert weight normalization parameters to actual weight.

    W = g * (v / ||v||)

    Args:
        weight_g: Magnitude [out_channels, 1, 1]
        weight_v: Direction [out_channels, in_channels, kernel_size]

    Returns:
        Weight tensor with same shape as weight_v
    """
    # Compute norm of v along dims 1 and 2 (in_channels, kernel_size)
    v_flat = weight_v.reshape(weight_v.shape[0], -1)
    v_norm = np.linalg.norm(v_flat, axis=1, keepdims=True)

    # Reshape norm to match weight_g shape for broadcasting
    v_norm = v_norm.reshape(weight_g.shape)

    # W = g * (v / ||v||)
    weight = (weight_g / (v_norm + 1e-12)) * weight_v

    return weight


def convert_hf_to_mlx(hf_model) -> dict[str, mx.array]:
    """Convert HuggingFace EnCodec model to MLX weights.

    Args:
        hf_model: HuggingFace EncodecModel

    Returns:
        Dictionary of MLX arrays
    """

    mlx_weights = {}

    # Get state dict
    state_dict = {}
    for name, param in hf_model.named_parameters():
        state_dict[name] = param.detach().cpu().numpy()
    for name, buffer in hf_model.named_buffers():
        state_dict[name] = buffer.detach().cpu().numpy()

    # Process each weight
    processed = set()

    # Find all weight-normalized convolutions
    weight_norm_bases = set()
    for key in state_dict:
        if "parametrizations.weight.original0" in key:
            base = key.replace(".parametrizations.weight.original0", "")
            weight_norm_bases.add(base)

    # Convert weight-normalized convolutions
    for base in weight_norm_bases:
        g_key = f"{base}.parametrizations.weight.original0"
        v_key = f"{base}.parametrizations.weight.original1"

        weight_g = state_dict[g_key]
        weight_v = state_dict[v_key]

        # Reconstruct weight
        weight = weight_norm_to_weight(weight_g, weight_v)

        # Check if this is a decoder upsample layer (transposed conv)
        # Decoder layers 3, 6, 9, 12 are the transposed convs
        is_transposed_conv = "decoder.layers." in base and any(
            f"decoder.layers.{idx}.conv" in base for idx in [3, 6, 9, 12]
        )

        if is_transposed_conv:
            # PyTorch ConvTranspose1d: [in_channels, out_channels, K]
            # MLX ConvTranspose1d: [out_channels, K, in_channels]
            weight = np.transpose(weight, (1, 2, 0))
        else:
            # PyTorch Conv1d: [out_channels, in_channels, K]
            # MLX Conv1d: [out_channels, K, in_channels]
            weight = np.transpose(weight, (0, 2, 1))

        # Map key to MLX format
        # The base already ends with ".conv" for HF, and we need ".conv.weight" for MLX
        mlx_key = f"{base}.weight"
        mlx_weights[mlx_key] = mx.array(weight.astype(np.float32))

        processed.add(g_key)
        processed.add(v_key)

        # Also get bias if present
        bias_key = f"{base}.bias"
        if bias_key in state_dict:
            mlx_weights[f"{base}.bias"] = mx.array(state_dict[bias_key].astype(np.float32))
            processed.add(bias_key)

    # Process LSTM weights
    # HF uses: lstm.weight_ih_l0, lstm.weight_hh_l0, lstm.bias_ih_l0, etc.
    # MLX uses: lstm.layers.0.Wx, lstm.layers.0.Wh, lstm.layers.0.bias
    for key in state_dict:
        if key in processed:
            continue

        if "lstm" not in key:
            continue

        value = state_dict[key]

        # Handle both layer 0 and layer 1
        for layer_idx in [0, 1]:
            suffix = f"_l{layer_idx}"
            if suffix not in key:
                continue

            # Map to MLX format with layers list
            if f"weight_ih{suffix}" in key:
                # Input-hidden weights: weight_ih_l0 -> layers.0.Wx
                mlx_key = key.replace(
                    f".lstm.weight_ih{suffix}",
                    f".lstm.{layer_idx}.Wx"
                )
                mlx_weights[mlx_key] = mx.array(value.astype(np.float32))
            elif f"weight_hh{suffix}" in key:
                # Hidden-hidden weights: weight_hh_l0 -> layers.0.Wh
                mlx_key = key.replace(
                    f".lstm.weight_hh{suffix}",
                    f".lstm.{layer_idx}.Wh"
                )
                mlx_weights[mlx_key] = mx.array(value.astype(np.float32))
            elif f"bias_ih{suffix}" in key:
                # Input bias: bias_ih_l0 -> layers.0.bias
                # NOTE: HF has separate bias_ih and bias_hh, we need to add them
                # For now, just use bias_ih (MLX typically uses bias = bias_ih)
                bias_hh_key = key.replace("bias_ih", "bias_hh")
                bias_ih = value
                bias_hh = state_dict.get(bias_hh_key, np.zeros_like(value))
                # MLX LSTM uses combined bias
                combined_bias = bias_ih + bias_hh
                mlx_key = key.replace(
                    f".lstm.bias_ih{suffix}",
                    f".lstm.{layer_idx}.bias"
                )
                mlx_weights[mlx_key] = mx.array(
                    combined_bias.astype(np.float32)
                )
            # Skip bias_hh (already combined above)

        processed.add(key)

    # Process quantizer codebook weights
    for key in state_dict:
        if key in processed:
            continue

        if "quantizer" in key and "codebook" in key:
            value = state_dict[key]
            mlx_weights[key] = mx.array(value.astype(np.float32))
            processed.add(key)

    return mlx_weights


def download_and_convert(
    model_name: str = "encodec_32khz",
    output_dir: str | Path | None = None,
) -> Path:
    """Download and convert EnCodec from HuggingFace.

    Args:
        model_name: One of encodec_24khz, encodec_32khz, encodec_48khz
        output_dir: Output directory

    Returns:
        Path to converted model
    """
    try:
        from transformers import EncodecModel
    except ImportError:
        raise ImportError("transformers required: pip install transformers")

    repos = {
        "encodec_24khz": "facebook/encodec_24khz",
        "encodec_32khz": "facebook/encodec_32khz",
        "encodec_48khz": "facebook/encodec_48khz",
    }

    if model_name not in repos:
        raise ValueError(f"Unknown model: {model_name}")

    if output_dir is None:
        output_dir = Path.home() / ".cache" / "mlx_audio" / "models"
    output_dir = Path(output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = output_dir / "model.safetensors"

    print(f"Loading {model_name} from HuggingFace...")
    hf_model = EncodecModel.from_pretrained(repos[model_name])

    print("Converting weights...")
    mlx_weights = convert_hf_to_mlx(hf_model)
    print(f"  Converted {len(mlx_weights)} parameters")

    print(f"Saving to {weights_path}...")
    mx.save_safetensors(str(weights_path), mlx_weights)

    # Save config - extract all values from HF config
    hf_config = hf_model.config
    config = {
        "sample_rate": hf_config.sampling_rate,
        "channels": hf_config.audio_channels,
        "num_filters": hf_config.num_filters,
        "num_codebooks": hf_config.num_quantizers,
        "codebook_size": hf_config.codebook_size,
        "codebook_dim": hf_config.codebook_dim,
        "ratios": hf_config.upsampling_ratios,
        "kernel_size": hf_config.kernel_size,
        "residual_kernel_size": hf_config.residual_kernel_size,
        "num_residual_layers": hf_config.num_residual_layers,
        "dilation_base": hf_config.dilation_growth_rate,
        "lstm_layers": hf_config.num_lstm_layers,
        "last_kernel_size": hf_config.last_kernel_size,
        "causal": hf_config.use_causal_conv,
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Done! Model saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="encodec_32khz",
                        choices=["encodec_24khz", "encodec_32khz", "encodec_48khz"])
    parser.add_argument("-o", "--output", help="Output directory")
    args = parser.parse_args()

    download_and_convert(args.model, args.output)
