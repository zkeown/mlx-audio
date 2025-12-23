"""Inference utilities for Banquet query-based source separation.

Provides chunked inference for processing long audio with overlap-add blending.
"""

from __future__ import annotations

from typing import Callable

import mlx.core as mx

from .banquet import Banquet, BanquetOutput
from .config import BanquetConfig


def apply_model(
    model: Banquet,
    mixture: mx.array,
    query_embedding: mx.array,
    *,
    segment: float | None = None,
    overlap: float = 0.25,
    split: bool = True,
    progress_callback: Callable[[float], None] | None = None,
) -> BanquetOutput:
    """Apply Banquet model with chunked inference.

    For long audio, the input is split into overlapping segments,
    processed independently, and blended using overlap-add.

    Args:
        model: Banquet model
        mixture: Input mixture [C, T] or [B, C, T]
        query_embedding: Pre-computed query embedding [768] or [B, 768]
        segment: Segment duration in seconds (None = 6.0 seconds default)
        overlap: Overlap ratio between segments (default: 0.25)
        split: Enable chunking (False = process entire audio at once)
        progress_callback: Optional progress callback function

    Returns:
        BanquetOutput with separated audio, spectrogram, and mask
    """
    # Add batch dimension if needed
    if mixture.ndim == 2:
        mixture = mixture[None, ...]
        squeeze_batch = True
    else:
        squeeze_batch = False

    if query_embedding.ndim == 1:
        query_embedding = query_embedding[None, ...]

    B, C, T = mixture.shape

    if segment is None:
        segment = 6.0  # Default 6 seconds

    segment_samples = int(segment * model.config.sample_rate)

    if not split or T <= segment_samples:
        # Process in single pass
        result = model(mixture, query_embedding)
    else:
        # Chunked processing with overlap-add
        result = _chunked_inference(
            model,
            mixture,
            query_embedding,
            segment_samples=segment_samples,
            overlap=overlap,
            progress_callback=progress_callback,
        )

    if squeeze_batch:
        result = BanquetOutput(
            audio=result.audio[0],
            spectrogram=result.spectrogram[0],
            mask=result.mask[0],
        )

    return result


def _chunked_inference(
    model: Banquet,
    mixture: mx.array,
    query_embedding: mx.array,
    *,
    segment_samples: int,
    overlap: float,
    progress_callback: Callable[[float], None] | None,
) -> BanquetOutput:
    """Process long audio in overlapping chunks.

    Uses overlap-add blending strategy to ensure smooth transitions.

    Args:
        model: Banquet model
        mixture: Input mixture [B, C, T]
        query_embedding: Query embedding [B, 768]
        segment_samples: Samples per segment
        overlap: Overlap ratio
        progress_callback: Optional progress callback

    Returns:
        BanquetOutput with separated audio, spectrogram, and mask
    """
    B, C, T = mixture.shape

    overlap_samples = int(segment_samples * overlap)
    stride = segment_samples - overlap_samples

    # Output buffer for audio
    out = mx.zeros((B, C, T))
    weight_sum = mx.zeros((B, C, T))

    # Triangular window for overlap-add
    weight = _create_weight_window(segment_samples)

    # Calculate number of chunks
    if T <= segment_samples:
        num_chunks = 1
    else:
        num_chunks = (T - overlap_samples + stride - 1) // stride

    # Store last mask and spectrogram (from final chunk for reference)
    last_mask = None
    last_spec = None

    for chunk_idx in range(num_chunks):
        offset = chunk_idx * stride
        chunk_end = min(offset + segment_samples, T)
        chunk_len = chunk_end - offset

        # Extract chunk
        chunk = mixture[:, :, offset:chunk_end]

        # Pad if needed
        if chunk_len < segment_samples:
            pad_amount = segment_samples - chunk_len
            chunk = mx.pad(chunk, [(0, 0), (0, 0), (0, pad_amount)])

        # Apply model
        chunk_result = model(chunk, query_embedding)

        # Trim audio to actual length if we padded
        chunk_audio = chunk_result.audio
        if chunk_len < segment_samples:
            chunk_audio = chunk_audio[:, :, :chunk_len]
            chunk_weight = weight[:chunk_len]
        else:
            chunk_weight = weight

        # Reshape weight for broadcasting: [1, 1, chunk_len]
        chunk_weight_3d = chunk_weight.reshape(1, 1, -1)

        # Weighted accumulation
        weighted_chunk = chunk_audio * chunk_weight_3d

        # Use slice assignment for accumulation
        out = out.at[:, :, offset:chunk_end].add(weighted_chunk)
        weight_sum = weight_sum.at[:, :, offset:chunk_end].add(
            mx.broadcast_to(chunk_weight_3d, weighted_chunk.shape)
        )

        # Store last mask/spec for output
        last_mask = chunk_result.mask
        last_spec = chunk_result.spectrogram

        # Progress callback
        if progress_callback:
            progress_callback((chunk_idx + 1) / num_chunks)

        # Evaluate to avoid memory buildup
        mx.eval(out, weight_sum)

    # Normalize by weights
    out = out / mx.maximum(weight_sum, 1e-8)

    return BanquetOutput(
        audio=out,
        spectrogram=last_spec,  # Last chunk spectrogram
        mask=last_mask,  # Last chunk mask
    )


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


def separate(
    model: Banquet,
    mixture: mx.array,
    query_mel: mx.array,
    *,
    segment: float | None = None,
    overlap: float = 0.25,
    split: bool = True,
    progress_callback: Callable[[float], None] | None = None,
) -> BanquetOutput:
    """Separate audio using query mel spectrogram.

    High-level API that encodes the query and runs separation.

    Args:
        model: Banquet model
        mixture: Input mixture [C, T] or [B, C, T]
        query_mel: Query mel spectrogram [1, n_mels, time] or [B, 1, n_mels, time]
        segment: Segment duration in seconds (None = 6.0 seconds default)
        overlap: Overlap ratio between segments
        split: Enable chunking
        progress_callback: Optional progress callback

    Returns:
        BanquetOutput with separated audio, spectrogram, and mask
    """
    # Add batch dimension to query if needed
    if query_mel.ndim == 3:
        query_mel = query_mel[None, ...]

    # Encode query
    query_embedding = model.encode_query(query_mel)

    # Run separation
    return apply_model(
        model,
        mixture,
        query_embedding,
        segment=segment,
        overlap=overlap,
        split=split,
        progress_callback=progress_callback,
    )


def prepare_query_mel(
    query_audio: mx.array,
    sample_rate: int = 44100,
    target_sample_rate: int = 32000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 320,
    win_length: int = 800,
    f_max: float = 16000.0,
) -> mx.array:
    """Prepare query mel spectrogram for PaSST encoder.

    Converts query audio to mel spectrogram in the format expected by PaSST.

    Args:
        query_audio: Query audio [C, T] or [T] (at sample_rate)
        sample_rate: Input sample rate
        target_sample_rate: PaSST expected sample rate (32kHz)
        n_mels: Number of mel bins
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        f_max: Maximum frequency for mel filterbank

    Returns:
        Mel spectrogram [1, 1, n_mels, time]
    """
    from mlx_audio.primitives import melspectrogram, resample

    # Convert to mono if stereo
    if query_audio.ndim == 2:
        query_audio = mx.mean(query_audio, axis=0)

    # Resample to 32kHz if needed
    if sample_rate != target_sample_rate:
        query_audio = resample(
            query_audio[None, :],
            sample_rate,
            target_sample_rate,
        )[0]

    # Compute mel spectrogram
    mel = melspectrogram(
        query_audio,
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        f_max=f_max,
    )

    # Add batch and channel dimensions: [n_mels, time] -> [1, 1, n_mels, time]
    mel = mel[None, None, ...]

    return mel
