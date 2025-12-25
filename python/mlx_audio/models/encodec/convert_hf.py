"""Weight conversion for EnCodec from HuggingFace to MLX.

Handles weight normalization decomposition and architecture mapping.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def compute_weight_from_norm(weight_g: np.ndarray, weight_v: np.ndarray) -> np.ndarray:
    """Reconstruct weight from weight normalization parameters.

    Weight normalization decomposes W = g * (v / ||v||)
    where g is the magnitude and v is the direction.

    Args:
        weight_g: Magnitude parameter (scalar per output channel)
        weight_v: Direction parameter (full weight shape)

    Returns:
        Reconstructed weight matrix
    """
    # weight_v shape: [out_channels, kernel_size, in_channels] for Conv1d
    # weight_g shape: [out_channels, 1, 1]

    # Normalize v along all but the first dimension
    v_norm = np.linalg.norm(weight_v.reshape(weight_v.shape[0], -1), axis=1, keepdims=True)
    v_norm = v_norm.reshape(weight_g.shape)

    # Reconstruct: W = g * (v / ||v||)
    weight = weight_g * weight_v / (v_norm + 1e-12)

    return weight


def convert_encodec_weights(
    hf_state_dict: dict[str, np.ndarray],
    prefix: str = "",
) -> dict[str, mx.array]:
    """Convert HuggingFace EnCodec weights to MLX format.

    Args:
        hf_state_dict: HuggingFace state dict (numpy arrays)
        prefix: Prefix to add to MLX keys

    Returns:
        MLX-compatible weight dictionary
    """
    mlx_weights = {}
    processed = set()

    # Group weights by base name to handle weight normalization
    weight_groups = {}
    for key in hf_state_dict:
        # Handle parametrizations.weight.original0/1 (weight normalization)
        if "parametrizations.weight.original0" in key:
            base = key.replace(".parametrizations.weight.original0", "")
            if base not in weight_groups:
                weight_groups[base] = {}
            weight_groups[base]["g"] = key
        elif "parametrizations.weight.original1" in key:
            base = key.replace(".parametrizations.weight.original1", "")
            if base not in weight_groups:
                weight_groups[base] = {}
            weight_groups[base]["v"] = key
        elif ".bias" in key and "lstm" not in key:
            # Regular bias
            base = key.replace(".bias", "")
            if base not in weight_groups:
                weight_groups[base] = {}
            weight_groups[base]["bias"] = key

    # Process weight-normalized convolutions
    for base, group in weight_groups.items():
        if "g" in group and "v" in group:
            weight_g = hf_state_dict[group["g"]]
            weight_v = hf_state_dict[group["v"]]

            # Reconstruct weight
            weight = compute_weight_from_norm(weight_g, weight_v)

            # Convert shape from PyTorch [out, in, K] to MLX Conv1d [out, K, in]
            weight = np.transpose(weight, (0, 2, 1))

            mlx_key = f"{prefix}{base}.conv.weight" if prefix else f"{base}.conv.weight"
            mlx_weights[mlx_key] = mx.array(weight)
            processed.add(group["g"])
            processed.add(group["v"])

        if "bias" in group:
            bias = hf_state_dict[group["bias"]]
            mlx_key = f"{prefix}{base}.conv.bias" if prefix else f"{base}.conv.bias"
            mlx_weights[mlx_key] = mx.array(bias)
            processed.add(group["bias"])

    # Process LSTM weights
    for key, value in hf_state_dict.items():
        if key in processed:
            continue

        if "lstm" in key:
            # LSTM weights need special handling
            # HF: lstm.weight_ih_l{layer}, lstm.weight_hh_l{layer}, etc.
            # MLX: lstm.Wx, lstm.Wh, lstm.bias

            # Map HF LSTM keys to MLX format
            mlx_key = key
            # Handle layer indices - MLX LSTM doesn't have multi-layer in one module
            # We'll need to split into separate LSTMs
            mlx_key = mlx_key.replace("weight_ih_l0", "Wx")
            mlx_key = mlx_key.replace("weight_hh_l0", "Wh")
            mlx_key = mlx_key.replace("bias_ih_l0", "bias")
            mlx_key = mlx_key.replace("weight_ih_l1", "Wx")  # For lstm2
            mlx_key = mlx_key.replace("weight_hh_l1", "Wh")
            mlx_key = mlx_key.replace("bias_ih_l1", "bias")
            # Skip hh biases (MLX combines them)
            if "bias_hh" in key:
                continue

            mlx_weights[f"{prefix}{mlx_key}" if prefix else mlx_key] = mx.array(value)
            processed.add(key)

    # Process quantizer codebook embeddings
    for key, value in hf_state_dict.items():
        if key in processed:
            continue

        if "quantizer" in key and "codebook" in key:
            # Map quantizer.layers.{i}.codebook.{attr}
            mlx_key = key
            mlx_weights[f"{prefix}{mlx_key}" if prefix else mlx_key] = mx.array(value)
            processed.add(key)

    return mlx_weights


def download_and_convert_encodec(
    model_name: str = "encodec_32khz",
    output_dir: str | Path | None = None,
) -> Path:
    """Download EnCodec from HuggingFace and convert to MLX format.

    Args:
        model_name: Model variant (encodec_24khz, encodec_32khz, encodec_48khz)
        output_dir: Output directory

    Returns:
        Path to converted model directory
    """
    try:
        from transformers import EncodecModel
    except ImportError:
        raise ImportError(
            "transformers is required for conversion. "
            "Install with: pip install transformers"
        )

    model_repos = {
        "encodec_24khz": "facebook/encodec_24khz",
        "encodec_32khz": "facebook/encodec_32khz",
        "encodec_48khz": "facebook/encodec_48khz",
    }

    if model_name not in model_repos:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_repos.keys())}")

    if output_dir is None:
        output_dir = Path.home() / ".cache" / "mlx_audio" / "models"
    output_dir = Path(output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    mlx_path = output_dir / "model.safetensors"

    print(f"Loading {model_name} from HuggingFace...")
    hf_model = EncodecModel.from_pretrained(model_repos[model_name])

    # Extract state dict as numpy
    state_dict = {}
    for name, param in hf_model.named_parameters():
        state_dict[name] = param.detach().cpu().numpy()
    for name, buffer in hf_model.named_buffers():
        state_dict[name] = buffer.detach().cpu().numpy()

    print(f"Converting {len(state_dict)} parameters...")
    mlx_weights = convert_encodec_weights(state_dict)

    print(f"Saving to {mlx_path}...")
    mx.save_safetensors(str(mlx_path), mlx_weights)

    # Save config
    config = {
        "sample_rate": hf_model.config.sampling_rate,
        "channels": hf_model.config.audio_channels,
        "num_filters": 32,  # Standard for all EnCodec models
        "num_codebooks": hf_model.config.num_quantizers,
        "codebook_size": hf_model.config.codebook_size,
        "codebook_dim": hf_model.config.codebook_dim,
        "ratios": [8, 5, 4, 2],  # Standard 32kHz ratios
        "lstm_layers": 2,
        "causal": True,
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Conversion complete! Saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert EnCodec HuggingFace weights to MLX")
    parser.add_argument(
        "-m", "--model",
        choices=["encodec_24khz", "encodec_32khz", "encodec_48khz"],
        default="encodec_32khz",
        help="Model variant to convert",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output directory",
    )

    args = parser.parse_args()
    download_and_convert_encodec(args.model, args.output)
