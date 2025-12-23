#!/usr/bin/env python3
"""Generate edge case fixtures for Swift parity tests.

This script generates fixtures for edge cases that should be handled
consistently between Python and Swift implementations:
- Very short audio (< 0.1 second)
- Silent audio (all zeros)
- Clipped audio (values at ±1.0)
- DC offset audio
- Single sample audio
- Maximum length audio

Usage:
    python generate_edge_case_fixtures.py --output-dir swift/Tests/Fixtures/EdgeCases
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    mx.random.seed(seed)


def save_arrays(path: Path, arrays: dict[str, mx.array]) -> None:
    """Save arrays to safetensors file."""
    mx.save_safetensors(str(path), arrays)
    print(f"  Saved: {path.name}")


def generate_audio_edge_cases(
    output_dir: Path,
    sample_rate: int = 44100,
) -> None:
    """Generate audio edge case fixtures."""
    print("Generating audio edge cases...")

    edge_cases = {}

    # 1. Very short audio (10ms)
    short_samples = int(0.01 * sample_rate)
    short_audio = np.sin(
        2 * np.pi * 440 * np.linspace(0, 0.01, short_samples)
    ).astype(np.float32)
    edge_cases["short_audio"] = mx.array(short_audio)
    edge_cases["short_audio_stereo"] = mx.array(
        np.stack([short_audio, short_audio])
    )

    # 2. Silent audio (all zeros)
    silent_audio = np.zeros(sample_rate, dtype=np.float32)  # 1 second
    edge_cases["silent_audio"] = mx.array(silent_audio)
    edge_cases["silent_audio_stereo"] = mx.array(
        np.stack([silent_audio, silent_audio])
    )

    # 3. Clipped audio (values at ±1.0)
    t = np.linspace(0, 1.0, sample_rate)
    loud_audio = np.sin(2 * np.pi * 440 * t) * 2.0  # Overdriven
    clipped_audio = np.clip(loud_audio, -1.0, 1.0).astype(np.float32)
    edge_cases["clipped_audio"] = mx.array(clipped_audio)

    # 4. DC offset audio
    dc_offset = 0.5
    dc_audio = (np.sin(2 * np.pi * 440 * t) * 0.3 + dc_offset).astype(np.float32)
    edge_cases["dc_offset_audio"] = mx.array(dc_audio)

    # 5. Single sample audio
    single_sample = np.array([0.5], dtype=np.float32)
    edge_cases["single_sample"] = mx.array(single_sample)

    # 6. Impulse (single 1.0 in silence)
    impulse = np.zeros(sample_rate, dtype=np.float32)
    impulse[sample_rate // 2] = 1.0
    edge_cases["impulse"] = mx.array(impulse)

    # 7. White noise
    noise = np.random.randn(sample_rate).astype(np.float32) * 0.5
    edge_cases["white_noise"] = mx.array(noise)

    # 8. Maximum amplitude
    max_amplitude = np.ones(sample_rate, dtype=np.float32)
    edge_cases["max_amplitude"] = mx.array(max_amplitude)

    # 9. Minimum amplitude (very quiet)
    min_amplitude = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 1e-6
    edge_cases["min_amplitude"] = mx.array(min_amplitude)

    # 10. Alternating samples (+1, -1)
    alternating = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(1000)], dtype=np.float32)
    edge_cases["alternating"] = mx.array(alternating)

    save_arrays(output_dir / "audio_edge_cases.safetensors", edge_cases)


def generate_mel_edge_cases(
    output_dir: Path,
    n_mels: int = 80,
) -> None:
    """Generate mel spectrogram edge case fixtures."""
    print("Generating mel spectrogram edge cases...")

    edge_cases = {}

    # 1. All zeros mel
    zero_mel = mx.zeros([1, n_mels, 100])
    edge_cases["zero_mel"] = zero_mel

    # 2. All ones mel
    ones_mel = mx.ones([1, n_mels, 100])
    edge_cases["ones_mel"] = ones_mel

    # 3. Very short mel (1 frame)
    short_mel = mx.random.normal([1, n_mels, 1])
    edge_cases["short_mel"] = short_mel

    # 4. Long mel (30 seconds worth)
    long_mel = mx.random.normal([1, n_mels, 3000])
    edge_cases["long_mel"] = long_mel

    # 5. Single channel with high values
    high_mel = mx.random.normal([1, n_mels, 100]) * 10.0
    edge_cases["high_mel"] = high_mel

    # 6. Single channel with very low values
    low_mel = mx.random.normal([1, n_mels, 100]) * 1e-6
    edge_cases["low_mel"] = low_mel

    save_arrays(output_dir / "mel_edge_cases.safetensors", edge_cases)


def generate_token_edge_cases(
    output_dir: Path,
    vocab_size: int = 51865,  # Whisper vocab size
) -> None:
    """Generate token sequence edge case fixtures."""
    print("Generating token edge cases...")

    edge_cases = {}

    # 1. Single token
    single_token = mx.array([[0]])
    edge_cases["single_token"] = single_token

    # 2. Empty-like (just special tokens)
    special_only = mx.array([[50257, 50362]])  # SOT, language
    edge_cases["special_only"] = special_only

    # 3. Maximum sequence length
    max_seq = mx.random.randint(low=0, high=vocab_size, shape=[1, 448])
    edge_cases["max_sequence"] = max_seq

    # 4. All same token
    same_token = mx.ones([1, 50], dtype=mx.int32) * 1000
    edge_cases["same_token"] = same_token

    # 5. Boundary tokens (0 and vocab_size-1)
    boundary = mx.array([[0, vocab_size - 1, 0, vocab_size - 1]])
    edge_cases["boundary_tokens"] = boundary

    save_arrays(output_dir / "token_edge_cases.safetensors", edge_cases)


def generate_embedding_edge_cases(
    output_dir: Path,
    embed_dim: int = 512,
) -> None:
    """Generate embedding edge case fixtures."""
    print("Generating embedding edge cases...")

    edge_cases = {}

    # 1. Zero embedding
    zero_embed = mx.zeros([1, embed_dim])
    edge_cases["zero_embedding"] = zero_embed

    # 2. Unit norm embedding
    unit_embed = mx.random.normal([1, embed_dim])
    unit_embed = unit_embed / mx.linalg.norm(unit_embed, axis=-1, keepdims=True)
    edge_cases["unit_embedding"] = unit_embed

    # 3. Very large embedding values
    large_embed = mx.random.normal([1, embed_dim]) * 100.0
    edge_cases["large_embedding"] = large_embed

    # 4. Very small embedding values
    small_embed = mx.random.normal([1, embed_dim]) * 1e-6
    edge_cases["small_embedding"] = small_embed

    # 5. All same value embedding
    same_embed = mx.ones([1, embed_dim]) * 0.5
    edge_cases["same_embedding"] = same_embed

    # 6. Sparse embedding (mostly zeros)
    sparse_embed = mx.zeros([1, embed_dim])
    sparse_indices = mx.array([[0, 100, 200, 300, 400]])
    sparse_embed = sparse_embed.at[0, 0].add(1.0)
    sparse_embed = sparse_embed.at[0, 100].add(1.0)
    sparse_embed = sparse_embed.at[0, 200].add(1.0)
    edge_cases["sparse_embedding"] = sparse_embed

    save_arrays(output_dir / "embedding_edge_cases.safetensors", edge_cases)


def generate_batch_edge_cases(
    output_dir: Path,
) -> None:
    """Generate batch-related edge case fixtures."""
    print("Generating batch edge cases...")

    edge_cases = {}

    # 1. Single item batch
    single_batch = mx.random.normal([1, 80, 100])
    edge_cases["single_batch"] = single_batch

    # 2. Large batch
    large_batch = mx.random.normal([16, 80, 100])
    edge_cases["large_batch"] = large_batch

    # 3. Odd batch size
    odd_batch = mx.random.normal([7, 80, 100])
    edge_cases["odd_batch"] = odd_batch

    save_arrays(output_dir / "batch_edge_cases.safetensors", edge_cases)


def main():
    parser = argparse.ArgumentParser(
        description="Generate edge case fixtures for Swift parity testing"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("swift/Tests/Fixtures/EdgeCases"),
        help="Output directory for fixtures",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    set_seed(args.seed)

    generate_audio_edge_cases(output_dir)
    print()

    generate_mel_edge_cases(output_dir)
    print()

    generate_token_edge_cases(output_dir)
    print()

    generate_embedding_edge_cases(output_dir)
    print()

    generate_batch_edge_cases(output_dir)
    print()

    print("All edge case fixtures generated successfully!")
    print(f"Fixtures saved to: {output_dir}")


if __name__ == "__main__":
    main()
