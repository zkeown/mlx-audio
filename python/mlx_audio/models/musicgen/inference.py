"""Inference utilities for MusicGen generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import mlx.core as mx

if TYPE_CHECKING:
    from mlx_audio.models.musicgen.model import MusicGen


@dataclass
class GenerationConfig:
    """Configuration for audio generation.

    Attributes:
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling (0 to disable)
        top_p: Nucleus sampling threshold (0 to disable)
        cfg_scale: Classifier-free guidance scale
        seed: Random seed for reproducibility
    """

    max_new_tokens: int = 500
    temperature: float = 1.0
    top_k: int = 250
    top_p: float = 0.0
    cfg_scale: float = 3.0
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
        # Get top-k values (MLX topk returns only values, not indices)
        top_values = mx.topk(logits, k=top_k, axis=-1)
        # Create mask for non-top-k positions (k-th largest value is at position -1)
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
        sorted_indices_to_remove = mx.concatenate([
            mx.zeros((logits.shape[0], 1), dtype=mx.bool_),
            sorted_indices_to_remove[:, :-1],
        ], axis=-1)

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


def apply_classifier_free_guidance(
    cond_logits: mx.array,
    uncond_logits: mx.array,
    cfg_scale: float,
) -> mx.array:
    """Apply classifier-free guidance.

    CFG formula: logits = uncond + cfg_scale * (cond - uncond)

    Args:
        cond_logits: Conditional logits [B, K, T, V]
        uncond_logits: Unconditional logits [B, K, T, V]
        cfg_scale: Guidance scale (1.0 = no guidance)

    Returns:
        Guided logits [B, K, T, V]
    """
    if cfg_scale == 1.0:
        return cond_logits

    return uncond_logits + cfg_scale * (cond_logits - uncond_logits)


def generate_tokens(
    model: "MusicGen",
    encoder_hidden_states: mx.array,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 250,
    top_p: float = 0.0,
    cfg_scale: float = 3.0,
    seed: int | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> mx.array:
    """Generate audio tokens autoregressively.

    Uses the delay pattern to generate multiple codebooks in parallel.

    Args:
        model: MusicGen model
        encoder_hidden_states: Text conditioning [B, S, D]
        max_new_tokens: Maximum tokens to generate per codebook
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling threshold
        cfg_scale: Classifier-free guidance scale
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

    # Prepare unconditional embeddings for CFG if needed
    # Note: projection happens in model.forward(), not here
    uncond_encoder_states = None
    if cfg_scale > 1.0:
        # Use zeros as unconditional conditioning (same shape as encoder states)
        uncond_encoder_states = mx.zeros_like(encoder_hidden_states)

    # Initialize with BOS tokens for all codebooks
    # Shape: [B, K, 1]
    current_tokens = mx.full(
        (batch_size, num_codebooks, 1),
        config.bos_token_id,
        dtype=mx.int32,
    )

    # KV cache for incremental decoding
    kv_cache = None
    uncond_kv_cache = None

    # Total steps including delay pattern overhead
    total_steps = max_new_tokens + num_codebooks - 1

    # Generate tokens step by step
    all_tokens = [current_tokens]

    for step in range(total_steps):
        # Report progress
        if progress_callback is not None:
            progress_callback(step / total_steps)

        # Compute position IDs
        position_ids = mx.array([[step]])
        position_ids = mx.broadcast_to(position_ids, (batch_size, 1))

        # Forward pass with conditioning
        logits, kv_cache = model.forward(
            current_tokens,
            encoder_hidden_states=encoder_hidden_states,
            kv_cache=kv_cache,
            position_ids=position_ids,
        )

        # Apply CFG if enabled
        if cfg_scale > 1.0 and uncond_encoder_states is not None:
            uncond_logits, uncond_kv_cache = model.forward(
                current_tokens,
                encoder_hidden_states=uncond_encoder_states,
                kv_cache=uncond_kv_cache,
                position_ids=position_ids,
            )
            logits = apply_classifier_free_guidance(
                logits, uncond_logits, cfg_scale
            )

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
                pad_tokens = mx.full((batch_size, 1, 1), config.pad_token_id, dtype=mx.int32)
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


def generate_audio(
    model: "MusicGen",
    encoder_hidden_states: mx.array,
    duration: float = 10.0,
    **kwargs,
) -> mx.array:
    """Generate audio waveform from text conditioning.

    Convenience function that generates tokens and decodes to audio.

    Args:
        model: MusicGen model with audio codec set
        encoder_hidden_states: Text conditioning [B, S, D]
        duration: Duration in seconds
        **kwargs: Additional arguments for generate_tokens

    Returns:
        Audio waveform [B, C, samples]
    """
    # Generate tokens
    max_new_tokens = int(duration * model.config.frame_rate)
    codes = generate_tokens(
        model,
        encoder_hidden_states,
        max_new_tokens=max_new_tokens,
        **kwargs,
    )

    # Decode to audio
    audio = model.decode_audio(codes)

    return audio
