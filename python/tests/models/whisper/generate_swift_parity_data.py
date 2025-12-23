#!/usr/bin/env python3
"""Generate reference data for Swift Whisper parity tests.

This script generates reference outputs from the Python Whisper implementation
that can be loaded and compared against in Swift tests.

Usage:
    python generate_swift_parity_data.py [--output-dir /tmp/whisper_parity]

The generated files can be used by swift/Tests/MLXAudioModelsTests/WhisperParityTests.swift
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_audio.models.whisper import Whisper, WhisperConfig
from mlx_audio.models.whisper.tokenizer import WhisperTokenizer


def generate_parity_data(output_dir: Path) -> None:
    """Generate all parity reference data."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Whisper parity data...")

    # Use tiny model for fast testing
    config = WhisperConfig.tiny()
    model = Whisper(config)

    # Initialize model with random weights for testing
    # (In production, would use from_pretrained)
    mx.eval(model.parameters())

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
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Generate test inputs
    np.random.seed(42)
    batch_size = 1
    mel_frames = 100
    seq_len = 5

    # Test mel spectrogram input [B, n_mels, T]
    mel_input = np.random.randn(batch_size, config.n_mels, mel_frames).astype(np.float32)
    np.save(output_dir / "mel_input.npy", mel_input)

    # Test token input [B, seq_len]
    token_input = np.random.randint(0, config.n_vocab, (batch_size, seq_len)).astype(np.int32)
    np.save(output_dir / "token_input.npy", token_input)

    # 1. Test encoder output
    print("  Generating encoder output...")
    mel_mx = mx.array(mel_input)
    encoder_output = model.encode(mel_mx)
    mx.eval(encoder_output)
    np.save(output_dir / "encoder_output.npy", np.array(encoder_output))

    # Save encoder output shape for verification
    encoder_shape = {"shape": list(encoder_output.shape)}
    with open(output_dir / "encoder_shape.json", "w") as f:
        json.dump(encoder_shape, f)

    # 2. Test decoder output (without cache)
    print("  Generating decoder output (no cache)...")
    tokens_mx = mx.array(token_input)
    logits, kv_cache = model.decode(tokens_mx, encoder_output)
    mx.eval(logits)
    np.save(output_dir / "decoder_logits_no_cache.npy", np.array(logits))

    # Save KV cache shapes
    kv_shapes = []
    for i, (k, v) in enumerate(kv_cache):
        mx.eval(k, v)
        kv_shapes.append({
            "layer": i,
            "key_shape": list(k.shape),
            "value_shape": list(v.shape),
        })
    with open(output_dir / "kv_cache_shapes.json", "w") as f:
        json.dump(kv_shapes, f, indent=2)

    # 3. Test decoder output with cache (incremental decoding)
    print("  Generating decoder output (with cache)...")
    # Feed single token with existing cache
    single_token = np.array([[token_input[0, -1]]], dtype=np.int32)
    single_token_mx = mx.array(single_token)
    logits_cached, _ = model.decode(single_token_mx, encoder_output, kv_cache)
    mx.eval(logits_cached)
    np.save(output_dir / "decoder_logits_with_cache.npy", np.array(logits_cached))

    # 4. Test full forward pass
    print("  Generating full forward pass output...")
    full_logits = model(mel_mx, tokens_mx)
    mx.eval(full_logits)
    np.save(output_dir / "full_forward_logits.npy", np.array(full_logits))

    # 5. Generate sinusoidal positional embeddings reference
    print("  Generating sinusoidal embeddings reference...")
    from mlx_audio.models.whisper.layers.encoder import sinusoids
    pos_emb = sinusoids(100, config.n_audio_state)
    mx.eval(pos_emb)
    np.save(output_dir / "sinusoids_100x384.npy", np.array(pos_emb))

    # 6. Tokenizer special tokens
    print("  Saving tokenizer info...")
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
        "all_language_tokens": tokenizer.all_language_tokens[:10],  # First 10
    }
    with open(output_dir / "tokenizer_info.json", "w") as f:
        json.dump(tokenizer_info, f, indent=2)

    # 7. Initial tokens for different configurations
    print("  Saving initial tokens...")
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

    # Save model weights for Swift to load
    print("  Saving model weights...")
    model.save_pretrained(output_dir / "model")

    print(f"\nParity data saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            print(f"  {f.relative_to(output_dir)}: {size:,} bytes")


def main():
    parser = argparse.ArgumentParser(description="Generate Whisper Swift parity data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/whisper_parity"),
        help="Output directory for parity data",
    )
    args = parser.parse_args()

    generate_parity_data(args.output_dir)


if __name__ == "__main__":
    main()
