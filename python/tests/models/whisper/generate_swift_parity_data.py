#!/usr/bin/env python3
"""Generate reference data for Swift Whisper parity tests.

This script generates reference outputs from the Python Whisper implementation
that can be loaded and compared against in Swift tests.

Usage:
    python generate_swift_parity_data.py --output-dir swift/Tests/Fixtures/Whisper

The generated files can be used by swift/Tests/MLXAudioModelsTests/WhisperParityTests.swift
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_audio.models.whisper import Whisper, WhisperConfig
from mlx_audio.models.whisper.tokenizer import WhisperTokenizer


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


def generate_sinusoids_fixtures(output_dir: Path, config: WhisperConfig) -> None:
    """Generate sinusoidal positional embedding fixtures."""
    print("Generating sinusoidal embeddings fixtures...")

    from mlx_audio.models.whisper.layers.encoder import sinusoids

    # Generate sinusoids for testing
    pos_emb = sinusoids(100, config.n_audio_state)
    mx.eval(pos_emb)

    save_arrays(
        output_dir / "sinusoids.safetensors",
        {
            "output": pos_emb,
            "length": mx.array([100]),
            "dim": mx.array([config.n_audio_state]),
        },
    )


def generate_encoder_fixtures(
    output_dir: Path,
    model: Whisper,
    config: WhisperConfig,
) -> None:
    """Generate encoder test fixtures."""
    print("Generating encoder fixtures...")

    # Generate random mel input [B, n_mels, T]
    batch_size = 1
    mel_frames = 100

    mel_input = mx.random.normal([batch_size, config.n_mels, mel_frames])

    # Run encoder
    encoder_output = model.encode(mel_input)
    mx.eval(encoder_output)

    # Save fixtures
    save_arrays(
        output_dir / "encoder.safetensors",
        {
            "input": mel_input,
            "output": encoder_output,
        },
    )


def generate_decoder_fixtures(
    output_dir: Path,
    model: Whisper,
    config: WhisperConfig,
) -> None:
    """Generate decoder test fixtures."""
    print("Generating decoder fixtures...")

    # Generate inputs
    batch_size = 1
    mel_frames = 100
    seq_len = 5

    mel_input = mx.random.normal([batch_size, config.n_mels, mel_frames])
    token_input = mx.random.randint(
        low=0, high=config.n_vocab, shape=[batch_size, seq_len]
    )

    # Run encoder to get audio features
    encoder_output = model.encode(mel_input)
    mx.eval(encoder_output)

    # Run decoder without cache
    logits_no_cache, kv_cache = model.decode(token_input, encoder_output)
    mx.eval(logits_no_cache)

    # Save decoder without cache fixtures
    save_arrays(
        output_dir / "decoder_no_cache.safetensors",
        {
            "mel_input": mel_input,
            "token_input": token_input,
            "encoder_output": encoder_output,
            "logits": logits_no_cache,
        },
    )

    # Run decoder with cache (single token)
    single_token = token_input[0:1, -1:]
    logits_with_cache, _ = model.decode(single_token, encoder_output, kv_cache)
    mx.eval(logits_with_cache)

    # Save decoder with cache fixtures
    save_arrays(
        output_dir / "decoder_with_cache.safetensors",
        {
            "single_token": single_token,
            "encoder_output": encoder_output,
            "logits": logits_with_cache,
        },
    )


def generate_full_model_fixtures(
    output_dir: Path,
    model: Whisper,
    config: WhisperConfig,
) -> None:
    """Generate full model test fixtures."""
    print("Generating full model fixtures...")

    # Generate inputs
    batch_size = 1
    mel_frames = 100
    seq_len = 5

    mel_input = mx.random.normal([batch_size, config.n_mels, mel_frames])
    token_input = mx.random.randint(
        low=0, high=config.n_vocab, shape=[batch_size, seq_len]
    )

    # Run full forward pass
    logits = model(mel_input, token_input)
    mx.eval(logits)

    # Save fixtures
    save_arrays(
        output_dir / "full_model.safetensors",
        {
            "mel_input": mel_input,
            "token_input": token_input,
            "logits": logits,
        },
    )


def generate_small_model_fixtures(output_dir: Path) -> None:
    """Generate fixtures for a small model with weights."""
    print("Generating small model fixtures with weights...")

    # Use tiny config
    config = WhisperConfig.tiny()
    model = Whisper(config)
    mx.eval(model.parameters())

    # Generate inputs
    batch_size = 1
    mel_frames = 100
    seq_len = 5

    mel_input = mx.random.normal([batch_size, config.n_mels, mel_frames])
    token_input = mx.random.randint(
        low=0, high=config.n_vocab, shape=[batch_size, seq_len]
    )

    # Run encoder
    encoder_output = model.encode(mel_input)
    mx.eval(encoder_output)

    # Run decoder
    logits, _ = model.decode(token_input, encoder_output)
    mx.eval(logits)

    # Save fixtures
    save_arrays(
        output_dir / "small_model.safetensors",
        {
            "mel_input": mel_input,
            "token_input": token_input,
            "encoder_output": encoder_output,
            "logits": logits,
        },
    )

    # Save weights
    weights = flatten_params(dict(model.parameters()))
    save_arrays(output_dir / "small_model_weights.safetensors", weights)

    # Save config
    config_dict = {
        "n_mels": config.n_mels,
        "n_audio_ctx": config.n_audio_ctx,
        "n_audio_state": config.n_audio_state,
        "n_audio_head": config.n_audio_head,
        "n_audio_layer": config.n_audio_layer,
        "n_text_ctx": config.n_text_ctx,
        "n_text_state": config.n_text_state,
        "n_text_head": config.n_text_head,
        "n_text_layer": config.n_text_layer,
        "n_vocab": config.n_vocab,
    }
    with open(output_dir / "small_model_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"  Saved: small_model_config.json")


def generate_tokenizer_fixtures(output_dir: Path) -> None:
    """Generate tokenizer reference fixtures."""
    print("Generating tokenizer fixtures...")

    tokenizer = WhisperTokenizer(multilingual=True)

    tokenizer_info = {
        "eot": tokenizer.eot,
        "sot": tokenizer.sot,
        "translate": tokenizer.translate,
        "transcribe": tokenizer.transcribe,
        "no_timestamps": tokenizer.no_timestamps,
        "no_speech": tokenizer.no_speech,
        "timestamp_begin": tokenizer.timestamp_begin,
        "timestamp_end": tokenizer.timestamp_end,
        "all_language_tokens": tokenizer.all_language_tokens[:10],
    }

    with open(output_dir / "tokenizer_info.json", "w") as f:
        json.dump(tokenizer_info, f, indent=2)
    print(f"  Saved: tokenizer_info.json")

    # Initial tokens for different configurations
    initial_tokens_info = {
        "transcribe_en_timestamps": tokenizer.get_initial_tokens(
            language="en", task="transcribe", timestamps=True
        ),
        "transcribe_en_no_timestamps": tokenizer.get_initial_tokens(
            language="en", task="transcribe", timestamps=False
        ),
        "translate_en": tokenizer.get_initial_tokens(
            language="en", task="translate", timestamps=True
        ),
    }

    with open(output_dir / "initial_tokens.json", "w") as f:
        json.dump(initial_tokens_info, f, indent=2)
    print(f"  Saved: initial_tokens.json")


def main():
    parser = argparse.ArgumentParser(
        description="Generate test fixtures for Swift Whisper parity testing"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("swift/Tests/Fixtures/Whisper"),
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

    # Create model with tiny config for layer tests
    config = WhisperConfig.tiny()
    model = Whisper(config)
    mx.eval(model.parameters())

    # Save main config
    config_dict = {
        "n_mels": config.n_mels,
        "n_audio_ctx": config.n_audio_ctx,
        "n_audio_state": config.n_audio_state,
        "n_audio_head": config.n_audio_head,
        "n_audio_layer": config.n_audio_layer,
        "n_text_ctx": config.n_text_ctx,
        "n_text_state": config.n_text_state,
        "n_text_head": config.n_text_head,
        "n_text_layer": config.n_text_layer,
        "n_vocab": config.n_vocab,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved config: config.json")
    print()

    # Generate layer fixtures
    generate_sinusoids_fixtures(output_dir, config)
    print()

    set_seed(args.seed)
    generate_encoder_fixtures(output_dir, model, config)
    print()

    set_seed(args.seed)
    generate_decoder_fixtures(output_dir, model, config)
    print()

    set_seed(args.seed)
    generate_tokenizer_fixtures(output_dir)
    print()

    if not args.layers_only:
        set_seed(args.seed)
        generate_full_model_fixtures(output_dir, model, config)
        print()

        set_seed(args.seed)
        generate_small_model_fixtures(output_dir)
        print()

    print("All Whisper fixtures generated successfully!")
    print(f"Fixtures saved to: {output_dir}")


if __name__ == "__main__":
    main()
