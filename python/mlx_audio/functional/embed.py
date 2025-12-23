"""Audio and text embedding API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import mlx.core as mx
    import numpy as np

    from mlx_audio.types.results import EmbeddingResult


@dataclass
class CLAPEmbeddingResult:
    """Result from CLAP embedding.

    Extends EmbeddingResult with audio-text similarity support.

    Attributes:
        audio_embeds: Audio embeddings [B, dim] or None
        text_embeds: Text embeddings [B, dim] or None
        similarity: Similarity matrix [B_audio, B_text] if both provided
        text_labels: Original text labels (for zero-shot)
        model_name: Name of the model used
        metadata: Additional metadata
    """

    audio_embeds: "mx.array | None" = None
    text_embeds: "mx.array | None" = None
    similarity: "mx.array | None" = None
    text_labels: list[str] | None = None
    model_name: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def vectors(self) -> "mx.array":
        """Get primary embedding vectors (audio if available, else text)."""
        if self.audio_embeds is not None:
            return self.audio_embeds
        if self.text_embeds is not None:
            return self.text_embeds
        raise ValueError("No embeddings available")

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self.vectors.shape[-1]

    def best_match(self, top_k: int = 1) -> list[str] | str:
        """Get best matching text label(s) for audio.

        Requires both audio and text embeddings with text_labels.

        Args:
            top_k: Number of top matches to return

        Returns:
            Best matching label(s)
        """
        if self.similarity is None or self.text_labels is None:
            raise ValueError("Need similarity matrix and text labels for best_match")

        import mlx.core as mx

        # Get top-k indices
        if self.similarity.ndim == 1:
            sim = self.similarity
        else:
            sim = self.similarity[0]  # First audio sample

        indices = mx.argsort(sim)[::-1][:top_k]
        indices = [int(i) for i in indices]

        matches = [self.text_labels[i] for i in indices]
        return matches[0] if top_k == 1 else matches

    def to_numpy(self) -> "np.ndarray":
        """Convert primary embeddings to NumPy array."""
        import numpy as np
        return np.array(self.vectors)

    def cosine_similarity(self, other: "CLAPEmbeddingResult") -> float:
        """Compute cosine similarity with another embedding."""
        import mlx.core as mx

        a = self.vectors
        b = other.vectors

        # Handle batched case
        if a.ndim > 1:
            a = a[0]
        if b.ndim > 1:
            b = b[0]

        a = a / mx.linalg.norm(a)
        b = b / mx.linalg.norm(b)
        return float(mx.sum(a * b))


def embed(
    audio: str | Path | "np.ndarray" | "mx.array" | None = None,
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
    import mlx.core as mx
    import numpy as np

    from mlx_audio.models.clap import CLAP
    from mlx_audio.hub.cache import get_cache

    if audio is None and text is None:
        raise ValueError("At least one of audio or text must be provided")

    # Load model
    cache = get_cache()
    clap = cache.get_model(model, CLAP)

    audio_embeds = None
    text_embeds = None
    text_labels = None

    # Process audio
    if audio is not None:
        audio_array, sr = _load_audio(audio, sample_rate)

        # Resample if needed
        if sr != clap.config.audio.sample_rate:
            from mlx_audio.primitives import resample
            audio_array = resample(audio_array, sr, clap.config.audio.sample_rate)

        # Ensure correct shape: [B, T]
        if audio_array.ndim == 1:
            audio_array = audio_array[None, :]
        elif audio_array.ndim == 2 and audio_array.shape[0] == 2:
            # Stereo to mono
            audio_array = mx.mean(audio_array, axis=0, keepdims=True)

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


def _load_audio(
    audio: str | Path | "np.ndarray" | "mx.array",
    sample_rate: int | None = None,
) -> tuple["mx.array", int]:
    """Load audio from file or array.

    Args:
        audio: Audio source
        sample_rate: Sample rate (required for array input)

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    import mlx.core as mx
    import numpy as np

    if isinstance(audio, (str, Path)):
        from mlx_audio.types.audio import load_audio
        audio_array, sr = load_audio(audio)
        if sample_rate is None:
            sample_rate = sr
    else:
        if isinstance(audio, np.ndarray):
            audio_array = mx.array(audio)
        else:
            audio_array = audio
        if sample_rate is None:
            sample_rate = 48000  # CLAP default

    return audio_array, sample_rate


def _tokenize_text(texts: list[str], max_length: int = 77) -> tuple["mx.array", "mx.array"]:
    """Tokenize text using RoBERTa tokenizer.

    Args:
        texts: List of text strings
        max_length: Maximum sequence length

    Returns:
        Tuple of (input_ids, attention_mask)
    """
    import mlx.core as mx

    try:
        from transformers import RobertaTokenizer
    except ImportError:
        raise ImportError(
            "transformers is required for text tokenization. "
            "Install with: pip install transformers"
        )

    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

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
