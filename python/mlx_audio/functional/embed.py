"""Audio and text embedding API."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx
    import numpy as np

from mlx_audio.constants import CLAP_SAMPLE_RATE
from mlx_audio.functional._audio import ensure_mono_batch, load_audio_input
from mlx_audio.types.results import CLAPEmbeddingResult

# Re-export for backward compatibility
__all__ = ["embed", "CLAPEmbeddingResult", "_tokenize_text"]


def embed(
    audio: str | Path | np.ndarray | mx.array | None = None,
    text: str | list[str] | None = None,
    *,
    model: str = "clap-htsat-fused",
    sample_rate: int | None = None,
    normalize: bool = True,
    return_similarity: bool = False,
    **kwargs,
) -> CLAPEmbeddingResult:
    """Compute audio and/or text embeddings using CLAP.

    CLAP (Contrastive Language-Audio Pretraining) encodes audio and text
    into a shared embedding space, enabling similarity search, zero-shot
    classification, and audio-text retrieval.

    Args:
        audio: Path to audio file, numpy array [C, T] or [T], or MLX array
        text: Text string or list of text strings
        model: Model name or path (default: "clap-htsat-fused")
        sample_rate: Audio sample rate (inferred from file if not provided)
        normalize: L2 normalize embeddings (default: True)
        return_similarity: Compute audio-text similarity matrix
        **kwargs: Additional model parameters

    Returns:
        CLAPEmbeddingResult with embeddings and optional similarity

    Examples:
        # Audio embedding only
        >>> result = embed(audio="dog_bark.wav")
        >>> result.audio_embeds  # [1, 512]

        # Text embedding only
        >>> result = embed(text="a dog barking loudly")
        >>> result.text_embeds  # [1, 512]

        # Zero-shot classification
        >>> result = embed(
        ...     audio="sound.wav",
        ...     text=["dog barking", "cat meowing", "bird singing"],
        ...     return_similarity=True,
        ... )
        >>> result.best_match()  # "dog barking"

        # Batch text embeddings
        >>> result = embed(text=["music", "speech", "noise"])
        >>> result.text_embeds  # [3, 512]
    """

    from mlx_audio.hub.cache import get_cache
    from mlx_audio.models.clap import CLAP

    if audio is None and text is None:
        from mlx_audio.exceptions import ConfigurationError
        raise ConfigurationError("At least one of audio or text must be provided")

    # Load model
    cache = get_cache()
    clap = cache.get_model(model, CLAP)

    audio_embeds = None
    text_embeds = None
    text_labels = None

    # Process audio
    if audio is not None:
        # Load and preprocess audio using shared utility
        audio_array, sr = load_audio_input(
            audio,
            sample_rate=sample_rate,
            default_sample_rate=CLAP_SAMPLE_RATE,
        )

        # Resample if needed
        if sr != clap.config.audio.sample_rate:
            from mlx_audio.primitives import resample
            audio_array = resample(audio_array, sr, clap.config.audio.sample_rate)

        # Ensure mono with batch dimension [B, T]
        audio_array = ensure_mono_batch(audio_array)

        # Encode audio
        audio_embeds = clap.encode_audio(audio_array, normalize=normalize)

    # Process text
    if text is not None:
        if isinstance(text, str):
            text = [text]
        text_labels = text

        # Tokenize text
        input_ids, attention_mask = _tokenize_text(text)

        # Encode text
        text_embeds = clap.encode_text(
            input_ids,
            attention_mask=attention_mask,
            normalize=normalize,
        )

    # Compute similarity if requested
    similarity = None
    if return_similarity and audio_embeds is not None and text_embeds is not None:
        similarity = clap.similarity(audio_embeds, text_embeds)

    return CLAPEmbeddingResult(
        audio_embeds=audio_embeds,
        text_embeds=text_embeds,
        similarity=similarity,
        text_labels=text_labels,
        model_name=model,
        metadata={
            "sample_rate": clap.config.audio.sample_rate,
            "projection_dim": clap.config.projection_dim,
        },
    )


def _get_roberta_tokenizer():
    """Get cached RoBERTa tokenizer (lazy-loaded on first call)."""
    try:
        from transformers import RobertaTokenizer
    except ImportError:
        from mlx_audio.exceptions import TokenizationError
        raise TokenizationError(
            "transformers is required for text tokenization. "
            "Install with: pip install transformers"
        )
    return RobertaTokenizer.from_pretrained("roberta-base")


# Module-level cached tokenizer (loaded once on first use)
_ROBERTA_TOKENIZER = None


def _tokenize_text(texts: list[str], max_length: int = 77) -> tuple[mx.array, mx.array]:
    """Tokenize text using RoBERTa tokenizer.

    Args:
        texts: List of text strings
        max_length: Maximum sequence length

    Returns:
        Tuple of (input_ids, attention_mask)
    """
    import mlx.core as mx

    global _ROBERTA_TOKENIZER
    if _ROBERTA_TOKENIZER is None:
        _ROBERTA_TOKENIZER = _get_roberta_tokenizer()

    tokenizer = _ROBERTA_TOKENIZER

    # Tokenize
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )

    input_ids = mx.array(encoded["input_ids"])
    attention_mask = mx.array(encoded["attention_mask"])

    return input_ids, attention_mask
