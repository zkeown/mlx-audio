#!/usr/bin/env python3
"""Generate test fixtures for MusicGen Swift parity testing.

This script generates intermediate activations and outputs from the Python
MusicGen model that can be loaded in Swift tests to verify numerical parity.

Usage:
    python tests/generate_musicgen_fixtures.py --output-dir swift/Tests/Fixtures/MusicGen

The script generates:
    - config.json: Model configuration
    - delay_pattern.safetensors: Delay pattern apply/revert operations
    - codebook_embed.safetensors: Codebook embedding layer input/output
    - decoder_layer.safetensors: Single decoder block input/output
    - lm_head.safetensors: Language model head input/output
    - small_model.safetensors: Small model forward pass
    - *_weights.safetensors: Model weights for each component
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_audio.models.musicgen.config import MusicGenConfig
from mlx_audio.models.musicgen import MusicGen
from mlx_audio.models.musicgen.layers.embeddings import CodebookEmbeddings
from mlx_audio.models.musicgen.layers.transformer import MusicGenDecoder
from mlx_audio.models.musicgen.layers.lm_head import MusicGenLMHead, DelayPatternScheduler


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


def generate_delay_pattern_fixtures(output_dir: Path) -> None:
    """Generate delay pattern scheduler fixtures."""
    print("Generating delay pattern fixtures...")

    num_codebooks = 4
    pad_token_id = 2048

    scheduler = DelayPatternScheduler(
        num_codebooks=num_codebooks,
        pad_token_id=pad_token_id,
    )

    # Create input tokens [B, K, T]
    batch_size = 2
    seq_length = 16

    # Random token IDs (below codebook_size to be valid)
    input_ids = mx.random.randint(low=0, high=2048, shape=[batch_size, num_codebooks, seq_length])

    # Apply delay pattern
    delayed = scheduler.apply_delay_pattern(input_ids)

    # Revert delay pattern
    reverted = scheduler.revert_delay_pattern(delayed)

    # Save fixtures
    save_arrays(
        output_dir / "delay_pattern.safetensors",
        {
            "input_ids": input_ids,
            "delayed": delayed,
            "reverted": reverted,
        },
    )

    # Save config
    config = {
        "num_codebooks": num_codebooks,
        "pad_token_id": pad_token_id,
    }
    with open(output_dir / "delay_pattern_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: delay_pattern_config.json")


def generate_codebook_embed_fixtures(output_dir: Path, config: MusicGenConfig) -> None:
    """Generate codebook embedding fixtures."""
    print("Generating codebook embedding fixtures...")

    embeddings = CodebookEmbeddings(config)

    # Input: [B, K, T] token IDs
    batch_size = 2
    num_codebooks = config.num_codebooks
    seq_length = 16

    input_ids = mx.random.randint(
        low=0, high=config.codebook_size, shape=[batch_size, num_codebooks, seq_length]
    )
    position_ids = mx.broadcast_to(mx.arange(seq_length)[None, :], [batch_size, seq_length])

    # Forward pass
    output = embeddings(input_ids, position_ids)

    # Save fixtures
    save_arrays(
        output_dir / "codebook_embed.safetensors",
        {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "output": output,
        },
    )

    # Save weights
    weights = flatten_params(dict(embeddings.parameters()))
    save_arrays(output_dir / "codebook_embed_weights.safetensors", weights)


def generate_decoder_layer_fixtures(output_dir: Path, config: MusicGenConfig) -> None:
    """Generate decoder layer fixtures."""
    print("Generating decoder layer fixtures...")

    decoder = MusicGenDecoder(config)

    # Input: [B, T, D] hidden states
    batch_size = 2
    seq_length = 16
    hidden_size = config.hidden_size

    hidden_states = mx.random.normal([batch_size, seq_length, hidden_size])

    # Encoder hidden states for cross-attention
    text_length = 8
    encoder_hidden = mx.random.normal([batch_size, text_length, hidden_size])

    # Causal mask
    attention_mask = decoder.create_causal_mask(seq_length)

    # Forward pass (no KV cache)
    output, kv_cache = decoder(
        hidden_states,
        encoder_hidden_states=encoder_hidden,
        attention_mask=attention_mask,
        encoder_attention_mask=None,
        kv_cache=None,
    )

    # Save fixtures
    save_arrays(
        output_dir / "decoder_layer.safetensors",
        {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden,
            "attention_mask": attention_mask,
            "output": output,
        },
    )

    # Save weights
    weights = flatten_params(dict(decoder.parameters()))
    save_arrays(output_dir / "decoder_layer_weights.safetensors", weights)


def generate_lm_head_fixtures(output_dir: Path, config: MusicGenConfig) -> None:
    """Generate LM head fixtures."""
    print("Generating LM head fixtures...")

    lm_head = MusicGenLMHead(config)

    # Input: [B, T, D] hidden states
    batch_size = 2
    seq_length = 16
    hidden_size = config.hidden_size

    hidden_states = mx.random.normal([batch_size, seq_length, hidden_size])

    # Forward pass
    logits = lm_head(hidden_states)  # [B, K, T, V]

    # Save fixtures
    save_arrays(
        output_dir / "lm_head.safetensors",
        {
            "hidden_states": hidden_states,
            "logits": logits,
        },
    )

    # Save weights
    weights = flatten_params(dict(lm_head.parameters()))
    save_arrays(output_dir / "lm_head_weights.safetensors", weights)


def generate_small_model_fixtures(output_dir: Path) -> None:
    """Generate fixtures for a small MusicGen model."""
    print("Generating small model fixtures...")

    # Create a small config for faster testing
    config = MusicGenConfig(
        num_codebooks=2,
        codebook_size=512,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        text_hidden_size=128,
    )

    model = MusicGen(config)

    # Generate inputs
    batch_size = 2
    seq_length = 8
    text_length = 4

    # Input token IDs [B, K, T]
    input_ids = mx.random.randint(
        low=0, high=config.codebook_size, shape=[batch_size, config.num_codebooks, seq_length]
    )

    # Text conditioning [B, S, D]
    encoder_hidden_states = mx.random.normal([batch_size, text_length, config.text_hidden_size])

    # Forward pass
    logits = model(input_ids, encoder_hidden_states=encoder_hidden_states)

    # Evaluate
    mx.eval(logits)

    # Save fixtures
    save_arrays(
        output_dir / "small_model.safetensors",
        {
            "input_ids": input_ids,
            "encoder_hidden_states": encoder_hidden_states,
            "logits": logits,
        },
    )

    # Save weights
    weights = flatten_params(dict(model.parameters()))
    save_arrays(output_dir / "small_model_weights.safetensors", weights)

    # Save config
    with open(output_dir / "small_model_config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"  Saved: small_model_config.json")


def generate_text_projection_fixtures(output_dir: Path, config: MusicGenConfig) -> None:
    """Generate text projection fixtures."""
    print("Generating text projection fixtures...")

    import mlx.nn as nn

    projection = nn.Linear(config.text_hidden_size, config.hidden_size)

    # Input: [B, S, text_hidden_size]
    batch_size = 2
    seq_length = 8

    proj_input = mx.random.normal([batch_size, seq_length, config.text_hidden_size])

    # Forward pass
    output = projection(proj_input)

    # Save fixtures
    save_arrays(
        output_dir / "text_projection.safetensors",
        {
            "input": proj_input,
            "output": output,
        },
    )

    # Save weights
    weights = {
        "weight": projection.weight,
        "bias": projection.bias if hasattr(projection, "bias") else mx.zeros([config.hidden_size]),
    }
    save_arrays(output_dir / "text_projection_weights.safetensors", weights)


def main():
    parser = argparse.ArgumentParser(
        description="Generate test fixtures for Swift MusicGen parity testing"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("swift/Tests/Fixtures/MusicGen"),
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
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    # Set seed
    set_seed(args.seed)

    # Use a small config for layer fixtures
    small_config = MusicGenConfig(
        num_codebooks=2,
        codebook_size=512,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        text_hidden_size=128,
    )

    # Generate fixtures
    generate_delay_pattern_fixtures(output_dir)
    print()

    set_seed(args.seed)
    generate_codebook_embed_fixtures(output_dir, small_config)
    print()

    set_seed(args.seed)
    generate_text_projection_fixtures(output_dir, small_config)
    print()

    set_seed(args.seed)
    generate_decoder_layer_fixtures(output_dir, small_config)
    print()

    set_seed(args.seed)
    generate_lm_head_fixtures(output_dir, small_config)
    print()

    if not args.layers_only:
        set_seed(args.seed)
        generate_small_model_fixtures(output_dir)
        print()

    # Save default config
    with open(output_dir / "config.json", "w") as f:
        json.dump(small_config.to_dict(), f, indent=2)
    print(f"Saved: config.json")

    print()
    print("All MusicGen fixtures generated successfully!")
    print(f"Fixtures saved to: {output_dir}")


if __name__ == "__main__":
    main()
