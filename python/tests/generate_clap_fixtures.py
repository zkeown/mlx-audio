#!/usr/bin/env python3
"""Generate test fixtures for CLAP Swift parity testing.

This script generates intermediate activations and outputs from the Python
CLAP model that can be loaded in Swift tests to verify numerical parity.

Usage:
    python tests/generate_clap_fixtures.py --output-dir swift/Tests/Fixtures/CLAP

The script generates:
    - config.json: Model configuration
    - patch_embed.safetensors: PatchEmbed layer input/output
    - swin_block.safetensors: Single SwinTransformerBlock input/output
    - htsat_encoder.safetensors: Full HTSAT encoder input/output
    - roberta_encoder.safetensors: RoBERTa text encoder input/output
    - audio_projection.safetensors: Audio projection head input/output
    - text_projection.safetensors: Text projection head input/output
    - full_model.safetensors: Full model audio/text embeddings
    - *_weights.safetensors: Model weights for each component
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_audio.models.clap import CLAP
from mlx_audio.models.clap.config import CLAPConfig, CLAPAudioConfig, CLAPTextConfig


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    mx.random.seed(seed)


def save_arrays(path: Path, arrays: dict[str, mx.array]) -> None:
    """Save arrays to safetensors file."""
    mx.save_safetensors(str(path), arrays)
    print(f"  Saved: {path.name}")


def flatten_params(params: dict, prefix: str = "") -> dict[str, mx.array]:
    """Flatten nested parameter dict with dot-separated keys."""
    flat = {}
    for k, v in params.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_params(v, key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    flat.update(flatten_params(item, f"{key}.{i}"))
                else:
                    flat[f"{key}.{i}"] = item
        else:
            flat[key] = v
    return flat


def generate_patch_embed_fixtures(output_dir: Path, config: CLAPAudioConfig) -> None:
    """Generate PatchEmbed layer fixtures."""
    print("Generating PatchEmbed fixtures...")

    from mlx_audio.models.clap.layers.patch_embed import PatchEmbed

    # Create PatchEmbed without fusion for simpler testing
    patch_embed = PatchEmbed(
        patch_size=config.patch_size,
        patch_stride=config.patch_stride,
        in_chans=1,
        embed_dim=config.embed_dim,
        flatten=True,
        enable_fusion=False,
        img_size=(config.n_mels, config.spec_size),
    )

    # Generate input: [B, 1, F, T] mel spectrogram
    batch_size = 2
    n_mels = config.n_mels  # 64
    time_frames = config.spec_size  # 256

    mel_input = mx.random.normal([batch_size, 1, n_mels, time_frames])

    # Forward pass
    output = patch_embed(mel_input)

    # Save fixtures
    save_arrays(
        output_dir / "patch_embed.safetensors",
        {
            "input": mel_input,
            "output": output,
        },
    )

    # Save weights
    weights = flatten_params(dict(patch_embed.parameters()))
    save_arrays(output_dir / "patch_embed_weights.safetensors", weights)


def generate_swin_block_fixtures(output_dir: Path, config: CLAPAudioConfig) -> None:
    """Generate SwinTransformerBlock fixtures."""
    print("Generating SwinTransformerBlock fixtures...")

    from mlx_audio.models.clap.layers.swin_block import BasicLayer

    # Create a single BasicLayer (contains multiple SwinTransformerBlocks)
    dim = config.embed_dim  # 96
    depth = 2  # Use small depth for testing
    num_heads = config.num_heads[0]  # 4

    basic_layer = BasicLayer(
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        window_size=config.window_size,
        mlp_ratio=config.mlp_ratio,
        qkv_bias=config.qkv_bias,
        drop=config.drop_rate,
        attn_drop=config.attn_drop_rate,
        drop_path=0.0,  # No stochastic depth for testing
        downsample=False,  # No downsampling for simpler testing
    )

    # Input shape: [B, L, C] - flattened spatial dims
    # BasicLayer expects (x, H, W) where x is [B, L, C] and L = H * W
    batch_size = 2
    height = 16
    width = 64  # Must be divisible by window_size (8)

    # Flatten spatial: [B, H*W, C]
    block_input = mx.random.normal([batch_size, height * width, dim])

    # Forward pass - BasicLayer returns (output, new_H, new_W)
    output, out_h, out_w = basic_layer(block_input, height, width)

    # Save fixtures
    save_arrays(
        output_dir / "swin_block.safetensors",
        {
            "input": block_input,
            "output": output,
            "height": mx.array([height]),
            "width": mx.array([width]),
            "out_height": mx.array([out_h]),
            "out_width": mx.array([out_w]),
        },
    )

    # Save weights
    weights = flatten_params(dict(basic_layer.parameters()))
    save_arrays(output_dir / "swin_block_weights.safetensors", weights)


def generate_htsat_fixtures(output_dir: Path, config: CLAPAudioConfig) -> None:
    """Generate HTSAT encoder fixtures."""
    print("Generating HTSAT encoder fixtures...")

    from mlx_audio.models.clap.layers.htsat import HTSAT

    # Create a small HTSAT for testing
    small_config = CLAPAudioConfig(
        n_mels=64,
        spec_size=256,
        patch_size=4,
        patch_stride=(4, 4),
        embed_dim=48,  # Smaller for faster testing
        depths=(1, 1, 2, 1),  # Fewer blocks
        num_heads=(2, 4, 8, 16),
        window_size=4,
        hidden_size=384,
        enable_fusion=False,
    )

    htsat = HTSAT(small_config)

    # Input: [B, 1, F, T] mel spectrogram
    batch_size = 2
    mel_input = mx.random.normal([batch_size, 1, small_config.n_mels, small_config.spec_size])

    # Forward pass
    output = htsat(mel_input)

    # Save fixtures
    save_arrays(
        output_dir / "htsat_encoder.safetensors",
        {
            "input": mel_input,
            "output": output,
        },
    )

    # Save weights
    weights = flatten_params(dict(htsat.parameters()))
    save_arrays(output_dir / "htsat_encoder_weights.safetensors", weights)

    # Save small config
    small_config_dict = {
        "n_mels": small_config.n_mels,
        "spec_size": small_config.spec_size,
        "patch_size": small_config.patch_size,
        "patch_stride": list(small_config.patch_stride),
        "embed_dim": small_config.embed_dim,
        "depths": list(small_config.depths),
        "num_heads": list(small_config.num_heads),
        "window_size": small_config.window_size,
        "hidden_size": small_config.hidden_size,
        "enable_fusion": small_config.enable_fusion,
    }
    with open(output_dir / "htsat_config.json", "w") as f:
        json.dump(small_config_dict, f, indent=2)
    print(f"  Saved: htsat_config.json")


def generate_roberta_fixtures(output_dir: Path, config: CLAPTextConfig) -> None:
    """Generate RoBERTa text encoder fixtures."""
    print("Generating RoBERTa encoder fixtures...")

    from mlx_audio.models.clap.layers.text_encoder import CLAPTextEncoder

    # Create a smaller text encoder for testing
    small_config = CLAPTextConfig(
        vocab_size=1000,  # Smaller vocab
        hidden_size=256,
        num_hidden_layers=2,  # Fewer layers
        num_attention_heads=4,
        intermediate_size=512,
    )

    projection_dim = 256
    text_encoder = CLAPTextEncoder(small_config, projection_dim)

    # Input: token IDs and attention mask
    batch_size = 2
    seq_length = 32

    # Generate random token IDs (avoiding special tokens at edges)
    input_ids = mx.random.randint(
        low=4, high=small_config.vocab_size, shape=[batch_size, seq_length]
    )
    attention_mask = mx.ones([batch_size, seq_length], dtype=mx.int32)

    # Forward pass
    output = text_encoder(input_ids, attention_mask=attention_mask, normalize=True)

    # Save fixtures
    save_arrays(
        output_dir / "roberta_encoder.safetensors",
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output": output,
        },
    )

    # Save weights
    weights = flatten_params(dict(text_encoder.parameters()))
    save_arrays(output_dir / "roberta_encoder_weights.safetensors", weights)

    # Save small config
    small_config_dict = {
        "vocab_size": small_config.vocab_size,
        "hidden_size": small_config.hidden_size,
        "num_hidden_layers": small_config.num_hidden_layers,
        "num_attention_heads": small_config.num_attention_heads,
        "intermediate_size": small_config.intermediate_size,
        "projection_dim": projection_dim,
    }
    with open(output_dir / "roberta_config.json", "w") as f:
        json.dump(small_config_dict, f, indent=2)
    print(f"  Saved: roberta_config.json")


def generate_projection_fixtures(output_dir: Path) -> None:
    """Generate projection head fixtures."""
    print("Generating projection head fixtures...")

    from mlx_audio.models.clap.model import CLAPProjection

    in_dim = 384
    out_dim = 256

    projection = CLAPProjection(in_dim, out_dim)

    # Input
    batch_size = 2
    proj_input = mx.random.normal([batch_size, in_dim])

    # Forward pass
    output = projection(proj_input)

    # Save fixtures
    save_arrays(
        output_dir / "audio_projection.safetensors",
        {
            "input": proj_input,
            "output": output,
        },
    )

    # Save weights
    weights = flatten_params(dict(projection.parameters()))
    save_arrays(output_dir / "audio_projection_weights.safetensors", weights)


def generate_small_model_fixtures(output_dir: Path) -> None:
    """Generate fixtures for a small CLAP model."""
    print("Generating small model fixtures...")

    # Create small configs for faster testing
    audio_config = CLAPAudioConfig(
        n_mels=64,
        spec_size=128,  # Smaller time frames
        patch_size=4,
        patch_stride=(4, 4),
        embed_dim=48,
        depths=(1, 1, 1, 1),  # Minimal depth
        num_heads=(2, 4, 8, 8),
        window_size=4,
        hidden_size=192,
        enable_fusion=False,
    )

    text_config = CLAPTextConfig(
        vocab_size=1000,
        hidden_size=192,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=384,
    )

    config = CLAPConfig(
        audio=audio_config,
        text=text_config,
        projection_dim=128,
    )

    model = CLAP(config)
    model.eval()  # Disable dropout

    # Generate inputs
    batch_size = 2

    # Audio input: [B, 1, F, T]
    audio_input = mx.random.normal([batch_size, 1, audio_config.n_mels, audio_config.spec_size])

    # Text input
    seq_length = 16
    input_ids = mx.random.randint(low=4, high=text_config.vocab_size, shape=[batch_size, seq_length])
    attention_mask = mx.ones([batch_size, seq_length], dtype=mx.int32)

    # Forward pass
    result = model(
        audio=audio_input,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    # Evaluate
    mx.eval(result["audio_embeds"])
    mx.eval(result["text_embeds"])
    mx.eval(result["logits_per_audio"])

    # Save fixtures
    save_arrays(
        output_dir / "small_model.safetensors",
        {
            "audio_input": audio_input,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio_embeds": result["audio_embeds"],
            "text_embeds": result["text_embeds"],
            "logits_per_audio": result["logits_per_audio"],
        },
    )

    # Save weights
    weights = flatten_params(dict(model.parameters()))
    save_arrays(output_dir / "small_model_weights.safetensors", weights)

    # Save config
    with open(output_dir / "small_model_config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"  Saved: small_model_config.json")


def generate_full_model_fixtures(output_dir: Path, use_pretrained: bool = False) -> None:
    """Generate full model fixtures."""
    print("Generating full model fixtures...")

    if use_pretrained:
        print("  Loading pretrained model...")
        try:
            from mlx_audio.hub import get_cache
            cache = get_cache()
            model = cache.get_model("clap-htsat-fused", CLAP)
            config = model.config
        except Exception as e:
            print(f"  Could not load pretrained model: {e}")
            print("  Falling back to random weights...")
            use_pretrained = False

    if not use_pretrained:
        print("  Creating model with random weights...")
        config = CLAPConfig()
        model = CLAP(config)

    model.eval()

    # Generate inputs
    batch_size = 1

    # Audio: [B, 1, F, T]
    n_mels = config.audio.n_mels
    spec_size = config.audio.spec_size
    audio_input = mx.random.normal([batch_size, 1, n_mels, spec_size])

    # Text
    seq_length = 32
    input_ids = mx.random.randint(low=4, high=1000, shape=[batch_size, seq_length])
    attention_mask = mx.ones([batch_size, seq_length], dtype=mx.int32)

    # Forward pass
    print("  Running forward pass...")
    result = model(
        audio=audio_input,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    mx.eval(result["audio_embeds"])
    mx.eval(result["text_embeds"])
    mx.eval(result["logits_per_audio"])

    # Save fixtures
    save_arrays(
        output_dir / "full_model.safetensors",
        {
            "audio_input": audio_input,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio_embeds": result["audio_embeds"],
            "text_embeds": result["text_embeds"],
            "logits_per_audio": result["logits_per_audio"],
        },
    )

    if not use_pretrained:
        # Save weights (only for random init, pretrained is too large)
        weights = flatten_params(dict(model.parameters()))
        save_arrays(output_dir / "full_model_weights.safetensors", weights)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"  Saved: config.json")


def main():
    parser = argparse.ArgumentParser(
        description="Generate test fixtures for Swift CLAP parity testing"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("swift/Tests/Fixtures/CLAP"),
        help="Output directory for fixtures",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
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

    # Set seed
    set_seed(args.seed)

    # Get default config for reference
    config = CLAPConfig()

    # Generate layer fixtures
    generate_patch_embed_fixtures(output_dir, config.audio)
    print()

    set_seed(args.seed)
    generate_swin_block_fixtures(output_dir, config.audio)
    print()

    set_seed(args.seed)
    generate_htsat_fixtures(output_dir, config.audio)
    print()

    set_seed(args.seed)
    generate_roberta_fixtures(output_dir, config.text)
    print()

    set_seed(args.seed)
    generate_projection_fixtures(output_dir)
    print()

    if not args.layers_only:
        set_seed(args.seed)
        generate_small_model_fixtures(output_dir)
        print()

        set_seed(args.seed)
        generate_full_model_fixtures(output_dir, use_pretrained=args.pretrained)
        print()

    print("All CLAP fixtures generated successfully!")
    print(f"Fixtures saved to: {output_dir}")


if __name__ == "__main__":
    main()
