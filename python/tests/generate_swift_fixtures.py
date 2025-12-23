#!/usr/bin/env python3
"""Generate test fixtures for Swift HTDemucs parity testing.

This script generates intermediate activations and outputs from the Python
HTDemucs model that can be loaded in Swift tests to verify numerical parity.

Usage:
    python tests/generate_swift_fixtures.py --output-dir swift/Tests/Fixtures

The script generates:
    - input.safetensors: Test input audio
    - dconv_*.safetensors: DConv layer inputs/outputs
    - encoder_*.safetensors: Encoder layer inputs/outputs
    - decoder_*.safetensors: Decoder layer inputs/outputs
    - transformer_*.safetensors: Transformer inputs/outputs
    - full_model.safetensors: Full model input/output
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def generate_layer_fixtures(output_dir: Path) -> None:
    """Generate fixtures for individual layer tests."""
    from mlx_audio.models.demucs.layers import DConv, HEncLayer, HDecLayer
    from mlx_audio.models.demucs.layers.transformer import (
        CrossTransformerEncoder,
        MultiheadAttention,
        MyTransformerEncoderLayer,
    )

    # Set seed for reproducibility
    mx.random.seed(42)

    print("Generating layer fixtures...")

    # --- DConv fixtures ---
    print("  DConv...")
    dconv = DConv(channels=48, depth=2, compress=8)
    dconv_input = mx.random.normal([2, 100, 48])  # [B, T, C] NLC format
    dconv_output = dconv(dconv_input)
    mx.save_safetensors(
        str(output_dir / "dconv.safetensors"),
        {
            "input": dconv_input,
            "output": dconv_output,
        },
    )
    # Save DConv weights
    dconv.save_weights(str(output_dir / "dconv_weights.safetensors"))

    # --- Frequency Encoder fixtures ---
    print("  Frequency Encoder...")
    freq_encoder = HEncLayer(
        chin=4,  # C*2 for CAC
        chout=48,
        kernel_size=8,
        stride=4,
        freq=True,
        dconv_depth=2,
        dconv_compress=8,
    )
    # [B, C, F, T] NCHW format
    freq_enc_input = mx.random.normal([1, 4, 512, 65])
    freq_enc_output = freq_encoder(freq_enc_input)
    mx.save_safetensors(
        str(output_dir / "freq_encoder.safetensors"),
        {
            "input": freq_enc_input,
            "output": freq_enc_output,
        },
    )
    freq_encoder.save_weights(str(output_dir / "freq_encoder_weights.safetensors"))

    # --- Time Encoder fixtures ---
    print("  Time Encoder...")
    time_encoder = HEncLayer(
        chin=2,
        chout=48,
        kernel_size=8,
        stride=4,
        freq=False,
        dconv_depth=2,
        dconv_compress=8,
    )
    # [B, C, T] NCL format
    time_enc_input = mx.random.normal([1, 2, 44100])
    time_enc_output = time_encoder(time_enc_input)
    mx.save_safetensors(
        str(output_dir / "time_encoder.safetensors"),
        {
            "input": time_enc_input,
            "output": time_enc_output,
        },
    )
    time_encoder.save_weights(str(output_dir / "time_encoder_weights.safetensors"))

    # --- Frequency Decoder fixtures ---
    print("  Frequency Decoder...")
    freq_decoder = HDecLayer(
        chin=48,
        chout=4,
        kernel_size=8,
        stride=4,
        freq=True,
        dconv_depth=2,
        dconv_compress=8,
        last=True,
    )
    freq_dec_input = mx.random.normal([1, 48, 128, 65])
    freq_dec_skip = mx.random.normal([1, 48, 128, 65])
    freq_dec_output, freq_dec_pre = freq_decoder(freq_dec_input, freq_dec_skip, length=65)
    mx.save_safetensors(
        str(output_dir / "freq_decoder.safetensors"),
        {
            "input": freq_dec_input,
            "skip": freq_dec_skip,
            "output": freq_dec_output,
            "pre": freq_dec_pre,
        },
    )
    freq_decoder.save_weights(str(output_dir / "freq_decoder_weights.safetensors"))

    # --- Time Decoder fixtures ---
    print("  Time Decoder...")
    time_decoder = HDecLayer(
        chin=48,
        chout=2,
        kernel_size=8,
        stride=4,
        freq=False,
        dconv_depth=2,
        dconv_compress=8,
        last=True,
    )
    time_dec_input = mx.random.normal([1, 48, 2756])
    time_dec_skip = mx.random.normal([1, 48, 2756])
    time_dec_output, time_dec_pre = time_decoder(time_dec_input, time_dec_skip, length=11025)
    mx.save_safetensors(
        str(output_dir / "time_decoder.safetensors"),
        {
            "input": time_dec_input,
            "skip": time_dec_skip,
            "length": mx.array([11025]),
            "output": time_dec_output,
            "pre": time_dec_pre,
        },
    )
    time_decoder.save_weights(str(output_dir / "time_decoder_weights.safetensors"))

    # --- MultiheadAttention fixtures ---
    print("  MultiheadAttention...")
    mha = MultiheadAttention(embed_dim=512, num_heads=8)
    mha_query = mx.random.normal([2, 100, 512])
    mha_key = mx.random.normal([2, 80, 512])
    mha_value = mha_key
    mha_output = mha(mha_query, mha_key, mha_value)
    mx.save_safetensors(
        str(output_dir / "multihead_attention.safetensors"),
        {
            "query": mha_query,
            "key": mha_key,
            "value": mha_value,
            "output": mha_output,
        },
    )
    mha.save_weights(str(output_dir / "multihead_attention_weights.safetensors"))

    # --- CrossTransformerEncoder fixtures ---
    print("  CrossTransformerEncoder...")
    transformer = CrossTransformerEncoder(
        dim=512,
        depth=5,
        heads=8,
        dim_feedforward=2048,
    )
    trans_freq = mx.random.normal([1, 512, 8, 16])  # [B, C, F, T]
    trans_time = mx.random.normal([1, 512, 100])    # [B, C, T]
    trans_freq_out, trans_time_out = transformer(trans_freq, trans_time)
    mx.save_safetensors(
        str(output_dir / "cross_transformer.safetensors"),
        {
            "freq_input": trans_freq,
            "time_input": trans_time,
            "freq_output": trans_freq_out,
            "time_output": trans_time_out,
        },
    )
    transformer.save_weights(str(output_dir / "cross_transformer_weights.safetensors"))

    print("Layer fixtures complete!")


def generate_full_model_fixtures(output_dir: Path, use_pretrained: bool = False) -> None:
    """Generate fixtures for full model test."""
    from mlx_audio.models.demucs import HTDemucs
    from mlx_audio.models.demucs.config import HTDemucsConfig

    print("Generating full model fixtures...")

    # Set seed for reproducibility
    mx.random.seed(42)

    if use_pretrained:
        print("  Loading pretrained model...")
        # This requires the model to be downloaded
        try:
            from mlx_audio.hub import get_cache
            cache = get_cache()
            model = cache.get_model("htdemucs_ft", HTDemucs)
        except Exception as e:
            print(f"  Could not load pretrained model: {e}")
            print("  Falling back to random weights...")
            use_pretrained = False

    if not use_pretrained:
        print("  Creating model with random weights...")
        config = HTDemucsConfig()
        model = HTDemucs(config)

    # Generate test input (1 second of audio)
    test_input = mx.random.normal([1, 2, 44100])

    print("  Running forward pass...")
    output = model(test_input)

    # Evaluate to ensure computation is complete
    mx.eval(output)

    print("  Saving fixtures...")
    mx.save_safetensors(
        str(output_dir / "full_model.safetensors"),
        {
            "input": test_input,
            "output": output,
        },
    )

    # Save model weights
    model.save_weights(str(output_dir / "full_model_weights.safetensors"))

    # Save config
    import json
    config_dict = {
        "sources": model.config.sources,
        "audio_channels": model.config.audio_channels,
        "samplerate": model.config.samplerate,
        "segment": model.config.segment,
        "channels": model.config.channels,
        "growth": model.config.growth,
        "depth": model.config.depth,
        "kernel_size": model.config.kernel_size,
        "stride": model.config.stride,
        "nfft": model.config.nfft,
        "hop_length": model.config.hop_length,
        "freq_emb": model.config.freq_emb,
        "t_depth": model.config.t_depth,
        "t_heads": model.config.t_heads,
        "t_dropout": model.config.t_dropout,
        "t_hidden_scale": model.config.t_hidden_scale,
        "bottom_channels": model.config.bottom_channels,
        "dconv_depth": model.config.dconv_depth,
        "dconv_comp": model.config.dconv_comp,
        "cac": model.config.cac,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print("Full model fixtures complete!")


def generate_stft_fixtures(output_dir: Path) -> None:
    """Generate fixtures for STFT/iSTFT parity testing."""
    from mlx_audio.primitives import stft, istft

    print("Generating STFT fixtures...")

    mx.random.seed(42)

    # Test input
    audio = mx.random.normal([2, 44100])  # 1 second stereo

    # STFT
    spec = stft(audio, n_fft=4096, hop_length=1024, pad_mode="edge")

    # Normalize like HTDemucs does
    import math
    spec_normalized = spec / math.sqrt(4096)

    # iSTFT
    reconstructed = istft(spec, hop_length=1024, length=44100)

    mx.save_safetensors(
        str(output_dir / "stft.safetensors"),
        {
            "audio": audio,
            "spec_real": mx.real(spec),
            "spec_imag": mx.imag(spec),
            "spec_normalized_real": mx.real(spec_normalized),
            "spec_normalized_imag": mx.imag(spec_normalized),
            "reconstructed": reconstructed,
        },
    )

    print("STFT fixtures complete!")


def generate_small_model_fixtures(output_dir: Path) -> None:
    """Generate fixtures for a small model (faster testing)."""
    from mlx_audio.models.demucs import HTDemucs
    from mlx_audio.models.demucs.config import HTDemucsConfig

    print("Generating small model fixtures...")

    mx.random.seed(42)

    # Small config for faster testing
    config = HTDemucsConfig(
        depth=2,
        channels=16,
        t_depth=1,
        bottom_channels=32,
        dconv_depth=1,
        segment=1.0,
    )
    model = HTDemucs(config)

    # Short test input
    test_input = mx.random.normal([1, 2, 22050])  # 0.5 seconds

    print("  Running forward pass...")
    output = model(test_input)
    mx.eval(output)

    print("  Saving fixtures...")
    mx.save_safetensors(
        str(output_dir / "small_model.safetensors"),
        {
            "input": test_input,
            "output": output,
        },
    )
    model.save_weights(str(output_dir / "small_model_weights.safetensors"))

    # Save config
    import json
    config_dict = {
        "sources": config.sources,
        "audio_channels": config.audio_channels,
        "samplerate": config.samplerate,
        "segment": config.segment,
        "channels": config.channels,
        "growth": config.growth,
        "depth": config.depth,
        "kernel_size": config.kernel_size,
        "stride": config.stride,
        "nfft": config.nfft,
        "hop_length": config.hop_length,
        "freq_emb": config.freq_emb,
        "t_depth": config.t_depth,
        "t_heads": config.t_heads,
        "t_dropout": config.t_dropout,
        "t_hidden_scale": config.t_hidden_scale,
        "bottom_channels": config.bottom_channels,
        "dconv_depth": config.dconv_depth,
        "dconv_comp": config.dconv_comp,
        "cac": config.cac,
    }
    with open(output_dir / "small_model_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print("Small model fixtures complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate test fixtures for Swift HTDemucs parity testing"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("swift/Tests/Fixtures/HTDemucs"),
        help="Output directory for fixtures",
    )
    parser.add_argument(
        "--layers-only",
        action="store_true",
        help="Only generate layer fixtures (faster)",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained weights for full model test",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    # Generate STFT fixtures first (no model needed)
    generate_stft_fixtures(output_dir)
    print()

    # Generate layer fixtures
    generate_layer_fixtures(output_dir)
    print()

    if not args.layers_only:
        # Generate small model fixtures (fast)
        generate_small_model_fixtures(output_dir)
        print()

        # Generate full model fixtures
        generate_full_model_fixtures(output_dir, use_pretrained=args.pretrained)
        print()

    print("All fixtures generated successfully!")
    print(f"Fixtures saved to: {output_dir}")


if __name__ == "__main__":
    main()
