"""Inference utilities for Parler-TTS generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import mlx.core as mx

if TYPE_CHECKING:
    from mlx_audio.models.tts.model import ParlerTTS


@dataclass
class TTSGenerationConfig:
    """Configuration for TTS generation.

    Attributes:
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling (0 to disable)
        top_p: Nucleus sampling threshold (0 to disable)
        seed: Random seed for reproducibility
    """

    max_new_tokens: int = 750  # ~10 seconds at 75 fps
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.0
    seed: int | None = None


def sample_next_token(
    logits: mx.array,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> mx.array:
    """Sample next token from logits.

    Args:
        logits: Logits [B, V] or [V]
        temperature: Sampling temperature
        top_k: Top-k sampling (0 to disable)
        top_p: Nucleus sampling threshold (0 to disable)

    Returns:
        Sampled token IDs [B] or scalar
    """
    # Handle single sample case
    squeeze = logits.ndim == 1
    if squeeze:
        logits = logits[None, :]

    # Apply temperature
    if temperature > 0:
        logits = logits / temperature
    else:
        # Greedy sampling
        tokens = mx.argmax(logits, axis=-1)
        return tokens.squeeze() if squeeze else tokens

    # Apply top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        # Get top-k values
        top_values = mx.topk(logits, k=top_k, axis=-1)
        # Create mask for non-top-k positions
        min_top_value = top_values[:, -1:]
        logits = mx.where(logits < min_top_value, float("-inf"), logits)

    # Apply top-p (nucleus) filtering
    if top_p > 0 and top_p < 1.0:
        sorted_logits = mx.sort(logits, axis=-1)[:, ::-1]
        sorted_probs = mx.softmax(sorted_logits, axis=-1)
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

        # Find cutoff index
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift right to keep at least one token
        sorted_indices_to_remove = mx.concatenate(
            [
                mx.zeros((logits.shape[0], 1), dtype=mx.bool_),
                sorted_indices_to_remove[:, :-1],
            ],
            axis=-1,
        )

        # Create threshold from sorted values
        sorted_logits = mx.where(
            sorted_indices_to_remove,
            float("-inf"),
            sorted_logits,
        )
        min_allowed = mx.min(sorted_logits, axis=-1, keepdims=True)
        logits = mx.where(logits < min_allowed, float("-inf"), logits)

    # Sample from distribution
    probs = mx.softmax(logits, axis=-1)
    tokens = mx.random.categorical(probs)

    return tokens.squeeze() if squeeze else tokens


def generate_tokens(
    model: "ParlerTTS",
    encoder_hidden_states: mx.array,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.0,
    seed: int | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> mx.array:
    """Generate audio tokens autoregressively.

    Uses the delay pattern to generate multiple codebooks in parallel.

    Args:
        model: Parler-TTS model
        encoder_hidden_states: Conditioning (already projected) [B, S, D]
        max_new_tokens: Maximum tokens to generate per codebook
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling threshold
        seed: Random seed
        progress_callback: Optional callback(progress: float)

    Returns:
        Generated codes [B, K, T]
    """
    config = model.config
    batch_size = encoder_hidden_states.shape[0]
    num_codebooks = config.num_codebooks

    # Set random seed if provided
    if seed is not None:
        mx.random.seed(seed)

    # Initialize with BOS tokens for all codebooks
    # Shape: [B, K, 1]
    current_tokens = mx.full(
        (batch_size, num_codebooks, 1),
        config.bos_token_id,
        dtype=mx.int32,
    )

    # KV cache for incremental decoding
    kv_cache = None

    # Total steps including delay pattern overhead
    total_steps = max_new_tokens + num_codebooks - 1

    # Generate tokens step by step
    all_tokens = [current_tokens]

    for step in range(total_steps):
        # Report progress
        if progress_callback is not None:
            progress_callback(step / total_steps)

        # Compute position offset for RoPE
        position_offset = step

        # Forward pass with conditioning
        logits, kv_cache = model.forward(
            current_tokens,
            encoder_hidden_states=encoder_hidden_states,
            kv_cache=kv_cache,
            position_offset=position_offset,
        )

        # Evaluate to materialize results
        mx.eval(logits)
        if kv_cache:
            mx.eval(kv_cache)

        # Sample next tokens for each codebook
        # logits shape: [B, K, 1, V]
        sampled_tokens = []

        for k in range(num_codebooks):
            # Only sample for codebooks that are "active" at this step
            # Codebook k becomes active at step k
            if step >= k:
                codebook_logits = logits[:, k, 0, :]  # [B, V]
                sampled = sample_next_token(
                    codebook_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                sampled_tokens.append(sampled[:, None, None])  # [B, 1, 1]
            else:
                # Use padding token for inactive codebooks
                pad_tokens = mx.full(
                    (batch_size, 1, 1), config.pad_token_id, dtype=mx.int32
                )
                sampled_tokens.append(pad_tokens)

        # Stack along codebook dimension: [B, K, 1]
        next_tokens = mx.concatenate(sampled_tokens, axis=1)

        all_tokens.append(next_tokens)
        current_tokens = next_tokens

    # Concatenate all tokens
    # Shape: [B, K, total_steps + 1]
    all_codes = mx.concatenate(all_tokens, axis=-1)

    # Remove BOS token and apply delay pattern reversion
    # Remove the first token (BOS)
    delayed_codes = all_codes[:, :, 1:]

    # Revert delay pattern to get aligned codes
    codes = model.delay_pattern.revert_delay_pattern(delayed_codes)

    # Trim to requested length
    codes = codes[:, :, :max_new_tokens]

    # Final progress update
    if progress_callback is not None:
        progress_callback(1.0)

    return codes


def generate_speech(
    model: "ParlerTTS",
    prompt_hidden_states: mx.array,
    description_hidden_states: mx.array | None = None,
    duration: float = 10.0,
    **kwargs,
) -> mx.array:
    """Generate speech waveform from text conditioning.

    Convenience function that generates tokens and decodes to audio.

    Args:
        model: Parler-TTS model with audio codec set
        prompt_hidden_states: Text prompt conditioning [B, S, D]
        description_hidden_states: Voice description conditioning [B, S, D]
        duration: Duration in seconds
        **kwargs: Additional arguments for generate_tokens

    Returns:
        Audio waveform [B, C, samples]
    """
    # Generate codes using model's generate method (handles conditioning projection)
    codes = model.generate(
        prompt_hidden_states=prompt_hidden_states,
        description_hidden_states=description_hidden_states,
        duration=duration,
        **kwargs,
    )

    # Decode to audio
    audio = model.decode_audio(codes)

    return audio
