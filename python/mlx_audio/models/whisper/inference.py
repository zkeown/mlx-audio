"""Whisper transcription inference.

Provides greedy and beam search decoding, chunked processing for
long audio, and timestamp extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from mlx_audio.models.whisper.model import Whisper
    from mlx_audio.models.whisper.tokenizer import WhisperTokenizer


@dataclass
class DecodingOptions:
    """Options for Whisper decoding.

    Attributes:
        language: Language code for transcription (None = auto-detect)
        task: "transcribe" or "translate"
        temperature: Sampling temperature (0 = greedy)
        beam_size: Beam search width (1 = greedy)
        best_of: Number of candidates for best-of-N sampling
        patience: Beam search patience factor
        length_penalty: Length normalization exponent
        max_tokens: Maximum tokens to generate
        without_timestamps: Disable timestamp prediction
        word_timestamps: Enable word-level timestamps
        suppress_blank: Suppress blank at beginning
        suppress_tokens: Token IDs to suppress
    """

    language: str | None = None
    task: str = "transcribe"

    # Sampling
    temperature: float = 0.0
    beam_size: int = 1
    best_of: int = 1
    patience: float = 1.0
    length_penalty: float = 1.0

    # Generation
    max_tokens: int = 448
    without_timestamps: bool = False
    word_timestamps: bool = False

    # Suppression
    suppress_blank: bool = True
    suppress_tokens: list[int] = field(default_factory=lambda: [-1])

    @property
    def is_greedy(self) -> bool:
        """Whether to use greedy decoding."""
        return self.temperature == 0.0 and self.beam_size == 1


@dataclass
class TranscriptionSegment:
    """A segment of transcribed text with timing."""

    text: str
    start: float
    end: float
    tokens: list[int] = field(default_factory=list)
    confidence: float = 0.0


def compute_log_mel_spectrogram(
    audio: mx.array,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    sample_rate: int = 16000,
) -> mx.array:
    """Compute log-mel spectrogram matching Whisper's preprocessing.

    Args:
        audio: Audio waveform [T] or [B, T]
        n_mels: Number of mel bins (80 for v1/v2, 128 for v3)
        n_fft: FFT window size
        hop_length: Hop length
        sample_rate: Sample rate

    Returns:
        Log-mel spectrogram [n_mels, T] or [B, n_mels, T]
    """
    from mlx_audio.primitives import melspectrogram

    # Compute mel spectrogram
    mel = melspectrogram(
        audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=0.0,
        fmax=8000.0,
    )

    # Convert to log scale (matching Whisper's exact formula)
    log_mel = mx.log10(mx.maximum(mel, 1e-10))
    log_mel = mx.maximum(log_mel, mx.max(log_mel) - 8.0)
    log_mel = (log_mel + 4.0) / 4.0  # Normalize to roughly [-1, 1]

    return log_mel


def pad_or_trim(audio: mx.array, length: int = 480000) -> mx.array:
    """Pad or trim audio to exact length.

    Args:
        audio: Audio waveform
        length: Target length (default: 30 seconds at 16kHz)

    Returns:
        Audio of exact length
    """
    if audio.shape[-1] > length:
        return audio[..., :length]
    elif audio.shape[-1] < length:
        pad_amount = length - audio.shape[-1]
        if audio.ndim == 1:
            return mx.pad(audio, [(0, pad_amount)])
        else:
            return mx.pad(audio, [(0, 0)] * (audio.ndim - 1) + [(0, pad_amount)])
    return audio


def greedy_decode(
    model: "Whisper",
    mel: mx.array,
    tokenizer: "WhisperTokenizer",
    options: DecodingOptions | None = None,
) -> list[int]:
    """Greedy decoding for transcription.

    Args:
        model: Whisper model
        mel: Log-mel spectrogram [B, n_mels, T] or [n_mels, T]
        tokenizer: Whisper tokenizer
        options: Decoding options

    Returns:
        List of token IDs
    """
    if options is None:
        options = DecodingOptions()

    # Handle unbatched input
    if mel.ndim == 2:
        mel = mel[None, :, :]

    # Encode audio
    audio_features = model.encode(mel)

    # Initial tokens
    initial_tokens = tokenizer.get_initial_tokens(
        language=options.language,
        task=options.task,
        timestamps=not options.without_timestamps,
    )

    tokens = mx.array([initial_tokens])
    all_tokens = list(initial_tokens)

    # KV cache for efficient decoding
    kv_cache = None

    for _ in range(options.max_tokens):
        # Get logits for next token
        logits, kv_cache = model.decode(tokens, audio_features, kv_cache)

        # Get next token (greedy)
        next_logits = logits[0, -1, :]

        # Apply temperature if not greedy
        if options.temperature > 0:
            next_logits = next_logits / options.temperature
            probs = mx.softmax(next_logits, axis=-1)
            # Sample from distribution
            next_token = int(mx.random.categorical(mx.log(probs)))
        else:
            next_token = int(mx.argmax(next_logits))

        # Check for end of text
        if next_token == tokenizer.eot:
            break

        all_tokens.append(next_token)
        tokens = mx.array([[next_token]])

        # Force evaluation to avoid memory buildup
        mx.eval(kv_cache)

    return all_tokens


def beam_search_decode(
    model: "Whisper",
    mel: mx.array,
    tokenizer: "WhisperTokenizer",
    options: DecodingOptions | None = None,
) -> list[int]:
    """Beam search decoding for transcription.

    Args:
        model: Whisper model
        mel: Log-mel spectrogram [B, n_mels, T] or [n_mels, T]
        tokenizer: Whisper tokenizer
        options: Decoding options

    Returns:
        List of token IDs from best beam
    """
    if options is None:
        options = DecodingOptions()

    # Handle unbatched input
    if mel.ndim == 2:
        mel = mel[None, :, :]

    beam_size = options.beam_size

    # Encode audio
    audio_features = model.encode(mel)

    # Expand for beam search
    audio_features = mx.repeat(audio_features, beam_size, axis=0)

    # Initial tokens
    initial_tokens = tokenizer.get_initial_tokens(
        language=options.language,
        task=options.task,
        timestamps=not options.without_timestamps,
    )

    # Initialize beams
    beams = [(initial_tokens, 0.0, None)]  # (tokens, score, kv_cache)

    for step in range(options.max_tokens):
        all_candidates = []

        for tokens, score, kv_cache in beams:
            if tokens[-1] == tokenizer.eot:
                all_candidates.append((tokens, score, kv_cache, True))
                continue

            # Get logits
            input_tokens = mx.array([[tokens[-1]]] if kv_cache else [tokens])
            logits, new_kv_cache = model.decode(
                input_tokens,
                audio_features[:1],  # Use first beam's features
                kv_cache,
            )

            # Get log probabilities
            log_probs = mx.log_softmax(logits[0, -1, :], axis=-1)

            # Get top-k tokens
            top_k = min(beam_size * 2, log_probs.shape[-1])
            top_indices = mx.argsort(log_probs)[-top_k:][::-1]

            for idx in top_indices[:beam_size]:
                token = int(idx)
                token_log_prob = float(log_probs[idx])
                new_tokens = tokens + [token]
                new_score = score + token_log_prob
                all_candidates.append((new_tokens, new_score, new_kv_cache, token == tokenizer.eot))

        # Select top beams
        all_candidates.sort(key=lambda x: x[1] / (len(x[0]) ** options.length_penalty), reverse=True)
        beams = [(tokens, score, cache) for tokens, score, cache, _ in all_candidates[:beam_size]]

        # Check if all beams are finished
        if all(tokens[-1] == tokenizer.eot for tokens, _, _ in beams):
            break

        mx.eval([cache for _, _, cache in beams if cache])

    # Return best beam
    best_tokens, _, _ = beams[0]
    return best_tokens


def transcribe_segment(
    model: "Whisper",
    mel: mx.array,
    tokenizer: "WhisperTokenizer",
    options: DecodingOptions | None = None,
) -> TranscriptionSegment:
    """Transcribe a single audio segment.

    Args:
        model: Whisper model
        mel: Log-mel spectrogram [n_mels, T]
        tokenizer: Whisper tokenizer
        options: Decoding options

    Returns:
        Transcription segment with text and timing
    """
    if options is None:
        options = DecodingOptions()

    # Decode
    if options.is_greedy:
        tokens = greedy_decode(model, mel, tokenizer, options)
    else:
        tokens = beam_search_decode(model, mel, tokenizer, options)

    # Decode text
    text = tokenizer.decode(tokens, skip_special_tokens=True)

    # Extract timestamps if present
    start = 0.0
    end = 30.0  # Default chunk length

    for i, token in enumerate(tokens):
        if tokenizer.is_timestamp(token):
            t = tokenizer.timestamp_to_seconds(token)
            if i == 0 or not tokenizer.is_timestamp(tokens[i - 1]):
                start = t
            else:
                end = t
                break

    return TranscriptionSegment(
        text=text.strip(),
        start=start,
        end=end,
        tokens=tokens,
    )


def transcribe_with_chunks(
    model: "Whisper",
    audio: mx.array,
    tokenizer: "WhisperTokenizer",
    options: DecodingOptions | None = None,
    chunk_length: float = 30.0,
    overlap: float = 5.0,
    progress_callback: Callable[[float], None] | None = None,
) -> list[TranscriptionSegment]:
    """Transcribe long audio using chunked processing.

    Args:
        model: Whisper model
        audio: Audio waveform [T] at 16kHz
        tokenizer: Whisper tokenizer
        options: Decoding options
        chunk_length: Chunk length in seconds
        overlap: Overlap between chunks in seconds
        progress_callback: Optional callback with progress [0, 1]

    Returns:
        List of transcription segments
    """
    if options is None:
        options = DecodingOptions()

    sample_rate = model.config.sample_rate
    n_mels = model.config.n_mels
    chunk_samples = int(chunk_length * sample_rate)
    overlap_samples = int(overlap * sample_rate)
    stride = chunk_samples - overlap_samples

    total_samples = audio.shape[-1]
    segments = []

    for start in range(0, total_samples, stride):
        # Extract chunk
        chunk = audio[start : start + chunk_samples]
        chunk = pad_or_trim(chunk, chunk_samples)

        # Compute mel spectrogram
        mel = compute_log_mel_spectrogram(
            chunk,
            n_mels=n_mels,
            n_fft=model.config.n_fft,
            hop_length=model.config.hop_length,
            sample_rate=sample_rate,
        )

        # Transcribe chunk
        segment = transcribe_segment(model, mel, tokenizer, options)

        # Adjust timestamps to absolute time
        time_offset = start / sample_rate
        segment.start += time_offset
        segment.end += time_offset

        # Only add if there's actual content
        if segment.text.strip():
            segments.append(segment)

        # Report progress
        if progress_callback:
            progress = min(1.0, (start + chunk_samples) / total_samples)
            progress_callback(progress)

        # Force evaluation to avoid memory buildup
        mx.eval(mel)

    # Merge overlapping segments
    merged = _merge_segments(segments)

    return merged


def _merge_segments(segments: list[TranscriptionSegment]) -> list[TranscriptionSegment]:
    """Merge overlapping transcription segments.

    Args:
        segments: List of segments with potential overlap

    Returns:
        Merged segments
    """
    if not segments:
        return []

    # Simple merging: combine segments that overlap
    merged = [segments[0]]

    for segment in segments[1:]:
        last = merged[-1]

        # Check for overlap
        if segment.start < last.end:
            # Merge: extend the end time and append text
            last.end = max(last.end, segment.end)
            if segment.text not in last.text:  # Avoid duplicates
                last.text = last.text.rstrip() + " " + segment.text.strip()
                last.tokens.extend(segment.tokens)
        else:
            merged.append(segment)

    return merged


def apply_model(
    model: "Whisper",
    audio: mx.array,
    tokenizer: "WhisperTokenizer",
    language: str | None = None,
    task: str = "transcribe",
    temperature: float = 0.0,
    beam_size: int = 1,
    word_timestamps: bool = False,
    progress_callback: Callable[[float], None] | None = None,
) -> "TranscriptionResult":
    """Apply Whisper model to audio.

    This is the main entry point for transcription.

    Args:
        model: Whisper model
        audio: Audio waveform [T] at 16kHz
        tokenizer: Whisper tokenizer
        language: Language code (None = auto-detect)
        task: "transcribe" or "translate"
        temperature: Sampling temperature
        beam_size: Beam search width
        word_timestamps: Enable word-level timestamps
        progress_callback: Optional progress callback

    Returns:
        TranscriptionResult with full transcription
    """
    from mlx_audio.types.results import TranscriptionResult, TranscriptionSegment as ResultSegment

    options = DecodingOptions(
        language=language,
        task=task,
        temperature=temperature,
        beam_size=beam_size,
        word_timestamps=word_timestamps,
    )

    # Detect language if needed
    detected_language = language
    if detected_language is None and model.config.is_multilingual:
        # Use first chunk for language detection
        sample_rate = model.config.sample_rate
        chunk = pad_or_trim(audio, model.config.n_samples)
        mel = compute_log_mel_spectrogram(
            chunk,
            n_mels=model.config.n_mels,
            n_fft=model.config.n_fft,
            hop_length=model.config.hop_length,
            sample_rate=sample_rate,
        )
        detected_language, _ = model.detect_language(mel, tokenizer)
        options.language = detected_language

    # Check if audio is longer than one chunk
    sample_rate = model.config.sample_rate
    chunk_samples = model.config.n_samples

    if audio.shape[-1] > chunk_samples:
        # Use chunked processing
        segments = transcribe_with_chunks(
            model, audio, tokenizer, options,
            progress_callback=progress_callback,
        )
    else:
        # Single chunk
        audio_padded = pad_or_trim(audio, chunk_samples)
        mel = compute_log_mel_spectrogram(
            audio_padded,
            n_mels=model.config.n_mels,
            n_fft=model.config.n_fft,
            hop_length=model.config.hop_length,
            sample_rate=sample_rate,
        )
        segment = transcribe_segment(model, mel, tokenizer, options)
        segments = [segment]

    # Build result
    result_segments = [
        ResultSegment(
            text=seg.text,
            start=seg.start,
            end=seg.end,
            confidence=seg.confidence,
        )
        for seg in segments
    ]

    full_text = " ".join(seg.text for seg in segments)

    return TranscriptionResult(
        text=full_text,
        segments=result_segments,
        language=detected_language,
    )
