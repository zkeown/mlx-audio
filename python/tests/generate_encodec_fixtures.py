#!/usr/bin/env python3
"""Generate test fixtures for EnCodec Swift parity tests.

This script generates safetensors fixtures that can be loaded by Swift tests
to verify numerical equivalence between Python and Swift implementations.

Usage:
    python tests/generate_encodec_fixtures.py --output-dir swift/Tests/Fixtures/EnCodec

The script generates:
    - config.json: Model configuration
    - encoder.safetensors: Encoder input/output
    - decoder.safetensors: Decoder input/output
    - quantizer.safetensors: RVQ input/output/codes
    - full_model.safetensors: Full model input/output
    - *_weights.safetensors: Model weights for each component
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_audio.models.encodec import EnCodec, EnCodecConfig


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    mx.random.seed(seed)


def save_arrays(path: Path, arrays: dict[str, mx.array]):
    """Save arrays to safetensors file."""
    mx.save_safetensors(str(path), arrays)
    print(f"  Saved: {path.name}")


def generate_encoder_fixtures(
    output_dir: Path,
    config: EnCodecConfig,
):
    """Generate encoder test fixtures."""
    print("Generating encoder fixtures...")

    from mlx_audio.models.encodec.layers.encoder import EnCodecEncoder

    encoder = EnCodecEncoder(config)

    # Generate random input
    batch_size = 2
    audio_length = 32000  # 1 second at 32kHz
    audio = mx.random.normal((batch_size, config.channels, audio_length))

    # Run encoder
    output = encoder(audio)

    # Save fixtures
    save_arrays(
        output_dir / "encoder.safetensors",
        {
            "input": audio,
            "output": output,
        },
    )

    # Save weights
    weights = dict(encoder.parameters())
    flat_weights = {}
    _flatten_dict(weights, flat_weights, prefix="")
    save_arrays(output_dir / "encoder_weights.safetensors", flat_weights)


def generate_decoder_fixtures(
    output_dir: Path,
    config: EnCodecConfig,
):
    """Generate decoder test fixtures."""
    print("Generating decoder fixtures...")

    from mlx_audio.models.encodec.layers.decoder import EnCodecDecoder

    decoder = EnCodecDecoder(config)

    # Generate random input (latent embeddings)
    batch_size = 2
    num_frames = 100
    embeddings = mx.random.normal((batch_size, num_frames, config.codebook_dim))

    # Run decoder
    output = decoder(embeddings)

    # Save fixtures
    save_arrays(
        output_dir / "decoder.safetensors",
        {
            "input": embeddings,
            "output": output,
        },
    )

    # Save weights
    weights = dict(decoder.parameters())
    flat_weights = {}
    _flatten_dict(weights, flat_weights, prefix="")
    save_arrays(output_dir / "decoder_weights.safetensors", flat_weights)


def generate_quantizer_fixtures(
    output_dir: Path,
    config: EnCodecConfig,
):
    """Generate quantizer test fixtures."""
    print("Generating quantizer fixtures...")

    from mlx_audio.models.encodec.layers.quantizer import (
        ResidualVectorQuantizer,
        VectorQuantizer,
    )

    # Single VQ
    vq = VectorQuantizer(config.codebook_size, config.codebook_dim)

    batch_size = 2
    num_frames = 100
    embeddings = mx.random.normal((batch_size, num_frames, config.codebook_dim))

    quantized, codes = vq(embeddings)

    save_arrays(
        output_dir / "vector_quantizer.safetensors",
        {
            "input": embeddings,
            "quantized": quantized,
            "codes": codes,
        },
    )

    # Save VQ weights
    vq_weights = dict(vq.parameters())
    flat_vq_weights = {}
    _flatten_dict(vq_weights, flat_vq_weights, prefix="")
    save_arrays(output_dir / "vector_quantizer_weights.safetensors", flat_vq_weights)

    # RVQ
    rvq = ResidualVectorQuantizer(
        config.num_codebooks,
        config.codebook_size,
        config.codebook_dim,
    )

    rvq_quantized, rvq_codes = rvq(embeddings)

    save_arrays(
        output_dir / "rvq.safetensors",
        {
            "input": embeddings,
            "quantized": rvq_quantized,
            "codes": rvq_codes,
        },
    )

    # Save RVQ weights
    rvq_weights = dict(rvq.parameters())
    flat_rvq_weights = {}
    _flatten_dict(rvq_weights, flat_rvq_weights, prefix="")
    save_arrays(output_dir / "rvq_weights.safetensors", flat_rvq_weights)


def generate_full_model_fixtures(
    output_dir: Path,
    config: EnCodecConfig,
):
    """Generate full model test fixtures."""
    print("Generating full model fixtures...")

    model = EnCodec(config)

    # Generate random audio input
    batch_size = 2
    audio_length = 32000  # 1 second
    audio = mx.random.normal((batch_size, config.channels, audio_length))

    # Run full model
    reconstructed, codes = model(audio)

    # Save fixtures
    save_arrays(
        output_dir / "full_model.safetensors",
        {
            "input": audio,
            "reconstructed": reconstructed,
            "codes": codes,
        },
    )

    # Save model weights
    weights = dict(model.parameters())
    flat_weights = {}
    _flatten_dict(weights, flat_weights, prefix="")
    save_arrays(output_dir / "full_model_weights.safetensors", flat_weights)


def generate_small_model_fixtures(
    output_dir: Path,
):
    """Generate fixtures for a smaller model (faster tests)."""
    print("Generating small model fixtures...")

    # Create a smaller config for faster testing
    config = EnCodecConfig(
        sample_rate=24000,
        channels=1,
        num_codebooks=2,
        codebook_size=512,
        codebook_dim=64,
        num_filters=16,
        num_residual_layers=1,
        ratios=[4, 4, 4],  # Smaller ratios for faster processing
        lstm_layers=0,  # No LSTM for simpler testing
    )

    model = EnCodec(config)

    # Generate random audio input
    batch_size = 1
    audio_length = 4096  # Short audio
    audio = mx.random.normal((batch_size, config.channels, audio_length))

    # Run full model
    reconstructed, codes = model(audio)

    # Save config
    config_path = output_dir / "small_model_config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"  Saved: {config_path.name}")

    # Save fixtures
    save_arrays(
        output_dir / "small_model.safetensors",
        {
            "input": audio,
            "reconstructed": reconstructed,
            "codes": codes,
        },
    )

    # Save model weights
    weights = dict(model.parameters())
    flat_weights = {}
    _flatten_dict(weights, flat_weights, prefix="")
    save_arrays(output_dir / "small_model_weights.safetensors", flat_weights)


def _flatten_dict(
    d: dict,
    out: dict,
    prefix: str = "",
):
    """Flatten nested dict with dot-separated keys."""
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten_dict(v, out, key)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    _flatten_dict(item, out, f"{key}.{i}")
                else:
                    out[f"{key}.{i}"] = item
        else:
            out[key] = v


def main():
    parser = argparse.ArgumentParser(description="Generate EnCodec test fixtures")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("swift/Tests/Fixtures/EnCodec"),
        help="Output directory for fixtures",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="encodec_24khz",
        choices=["encodec_24khz", "encodec_32khz", "encodec_48khz_stereo"],
        help="Config preset to use",
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set seed
    set_seed(args.seed)

    # Get config
    config = EnCodecConfig.from_name(args.config)

    # Save config
    config_path = args.output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Saved config: {config_path}")

    # Generate fixtures
    generate_encoder_fixtures(args.output_dir, config)
    generate_decoder_fixtures(args.output_dir, config)
    generate_quantizer_fixtures(args.output_dir, config)
    generate_full_model_fixtures(args.output_dir, config)
    generate_small_model_fixtures(args.output_dir)

    print(f"\nAll fixtures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
