"""Inference utilities for HTDemucs."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Union

import mlx.core as mx

from mlx_audio.models.demucs.model import HTDemucs

if TYPE_CHECKING:
    from mlx_audio.models.demucs.bag import BagOfModels

# Type alias for model parameter
ModelType = Union[HTDemucs, "BagOfModels"]


def apply_model(
    model: ModelType,
    mix: mx.array,
    *,
    segment: float | None = None,
    overlap: float = 0.25,
    shifts: int = 1,
    split: bool = True,
    progress_callback: Callable[[float], None] | None = None,
    **kwargs: Any,
) -> mx.array:
    """Apply HTDemucs model with chunked inference.

    For long audio, the input is split into overlapping segments,
    processed independently, and blended using overlap-add.

    Works with both single HTDemucs models and BagOfModels ensembles.

    Args:
        model: HTDemucs model or BagOfModels ensemble
        mix: Input mixture [C, T] or [B, C, T]
        segment: Segment duration in seconds (None = use model default)
        overlap: Overlap ratio between segments
        shifts: Number of random time shifts for augmentation
        split: Enable chunking (False = process entire audio at once)
        progress_callback: Optional progress callback function
        **kwargs: Additional arguments

    Returns:
        Separated stems [S, C, T] or [B, S, C, T]
    """
    # Add batch dimension if needed
    if mix.ndim == 2:
        mix = mix[None, ...]
        squeeze_batch = True
    else:
        squeeze_batch = False

    B, C, T = mix.shape
    _ = B, C  # Unused but documents expected shape

    if segment is None:
        segment = model.config.segment

    segment_samples = int(segment * model.config.samplerate)

    if not split or segment_samples >= T:
        # Process in single pass
        stems = model(mix)
    else:
        # Chunked processing with overlap-add
        stems = _chunked_inference(
            model,
            mix,
            segment_samples=segment_samples,
            overlap=overlap,
            progress_callback=progress_callback,
        )

    if squeeze_batch:
        stems = stems[0]

    return stems


def _chunked_inference(
    model: HTDemucs,
    mix: mx.array,
    *,
    segment_samples: int,
    overlap: float,
    progress_callback: Callable[[float], None] | None,
) -> mx.array:
    """Process long audio in overlapping chunks.

    Uses the same overlap-add blending strategy as the streaming processor
    to ensure identical outputs between batch and streaming inference.

    Args:
        model: HTDemucs model
        mix: Input mixture [B, C, T]
        segment_samples: Samples per segment
        overlap: Overlap ratio
        progress_callback: Optional progress callback

    Returns:
        Separated stems [B, S, C, T]
    """
    B, C, T = mix.shape
    S = model.config.num_sources

    overlap_samples = int(segment_samples * overlap)
    stride = segment_samples - overlap_samples

    # Output buffer
    out = mx.zeros((B, S, C, T))
    weight_sum = mx.zeros((B, S, C, T))

    # Triangular window for overlap-add
    weight = _create_weight_window(segment_samples)

    # Calculate number of chunks - ensure we cover all samples
    # The last chunk may extend past T but we'll handle padding
    num_chunks = 1 if segment_samples >= T else (T - overlap_samples + stride - 1) // stride

    for chunk_idx in range(num_chunks):
        offset = chunk_idx * stride
        chunk_end = min(offset + segment_samples, T)
        chunk_len = chunk_end - offset

        # Extract chunk
        chunk = mix[:, :, offset:chunk_end]

        # Pad if needed
        if chunk_len < segment_samples:
            pad_amount = segment_samples - chunk_len
            chunk = mx.pad(chunk, [(0, 0), (0, 0), (0, pad_amount)])

        # Apply model
        chunk_out = model(chunk)

        # Trim to actual length if we padded
        if chunk_len < segment_samples:
            chunk_out = chunk_out[:, :, :, :chunk_len]
            chunk_weight = weight[:chunk_len]
        else:
            chunk_weight = weight

        # Reshape weight for broadcasting: [1, 1, 1, chunk_len]
        chunk_weight_4d = chunk_weight.reshape(1, 1, 1, -1)

        # Weighted accumulation
        weighted_chunk = chunk_out * chunk_weight_4d

        # Use slice assignment for accumulation
        out = out.at[:, :, :, offset:chunk_end].add(weighted_chunk)
        weight_sum = weight_sum.at[:, :, :, offset:chunk_end].add(
            mx.broadcast_to(chunk_weight_4d, weighted_chunk.shape)
        )

        # Progress callback
        if progress_callback:
            progress_callback((chunk_idx + 1) / num_chunks)

        # Evaluate to avoid memory buildup
        mx.eval(out, weight_sum)

    # Normalize by weights
    out = out / mx.maximum(weight_sum, 1e-8)

    return out


def _create_weight_window(length: int, power: float = 1.0) -> mx.array:
    """Create triangular weight window for overlap-add.

    Args:
        length: Window length
        power: Power to raise the window to

    Returns:
        Weight window
    """
    half = length // 2
    ramp_up = mx.arange(1, half + 1, dtype=mx.float32)
    ramp_down = mx.arange(length - half, 0, -1, dtype=mx.float32)
    weight = mx.concatenate([ramp_up, ramp_down])
    weight = (weight / mx.max(weight)) ** power
    return weight
