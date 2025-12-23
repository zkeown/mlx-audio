"""Audio tagging API."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx
    import numpy as np

from mlx_audio.types.results import TaggingResult


def tag(
    audio: str | Path | "np.ndarray" | "mx.array",
    *,
    model: str = "clap-htsat-fused",
    tags: list[str] | None = None,
    threshold: float = 0.5,
    sample_rate: int | None = None,
    **kwargs,
) -> TaggingResult:
    """Tag audio with multiple labels (multi-label classification).

    Uses CLAP embeddings with zero-shot tagging via text similarity.
    For trained taggers, specify a model path or registry name.

    Args:
        audio: Path to audio file, numpy array [C, T] or [T], or MLX array
        model: Model name, path, or CLAP model for zero-shot tagging
        tags: Tag labels for zero-shot tagging (required for CLAP models)
        threshold: Probability threshold for active tags
        sample_rate: Audio sample rate (inferred from file if not provided)
        **kwargs: Additional model parameters

    Returns:
        TaggingResult with active tags and probabilities

    Examples:
        Zero-shot tagging:
        >>> result = tag(
        ...     "music.wav",
        ...     tags=["piano", "guitar", "drums", "vocals", "bass"]
        ... )
        >>> result.tags  # ["piano", "vocals"]
        >>> result.top_k(3)  # [("piano", 0.92), ("vocals", 0.85), ("guitar", 0.45)]

        With trained tagger:
        >>> result = tag("audio.wav", model="./audioset_tagger", threshold=0.3)
        >>> for t, prob in result.above_threshold():
        ...     print(f"{t}: {prob:.1%}")
    """
    import mlx.core as mx
    import numpy as np

    from mlx_audio.hub.cache import get_cache

    # Load and preprocess audio
    audio_array, sr = _load_audio(audio, sample_rate)

    # Ensure batch dimension
    if audio_array.ndim == 1:
        audio_array = audio_array[None, :]
    elif audio_array.ndim == 2 and audio_array.shape[0] == 2:
        # Stereo to mono
        audio_array = mx.mean(audio_array, axis=0, keepdims=True)

    # Check if this is a CLAP model (for zero-shot) or trained tagger
    if _is_clap_model(model):
        return _zero_shot_tag(
            audio_array, sr, model, tags, threshold, **kwargs
        )
    else:
        return _trained_tag(
            audio_array, sr, model, tags, threshold, **kwargs
        )


def _is_clap_model(model: str) -> bool:
    """Check if model name refers to a CLAP model."""
    clap_models = {"clap-htsat-fused", "clap-htsat-unfused"}
    return model in clap_models or "clap" in model.lower()


def _zero_shot_tag(
    audio: "mx.array",
    sample_rate: int,
    model: str,
    tags: list[str] | None,
    threshold: float,
    **kwargs,
) -> TaggingResult:
    """Perform zero-shot tagging using CLAP text-audio similarity."""
    import mlx.core as mx

    if tags is None or len(tags) == 0:
        raise ValueError(
            "tags must be provided for zero-shot tagging. "
            "Example: tag(audio, tags=['piano', 'guitar', 'drums'])"
        )

    from mlx_audio.models.clap import CLAP
    from mlx_audio.hub.cache import get_cache

    # Load CLAP model
    cache = get_cache()
    clap = cache.get_model(model, CLAP)

    # Resample if needed
    if sample_rate != clap.config.audio.sample_rate:
        from mlx_audio.primitives import resample
        audio = resample(audio, sample_rate, clap.config.audio.sample_rate)
        sample_rate = clap.config.audio.sample_rate

    # Encode audio
    audio_embeds = clap.encode_audio(audio, normalize=True)

    # Tokenize and encode tags
    from mlx_audio.functional.embed import _tokenize_text
    input_ids, attention_mask = _tokenize_text(tags)
    text_embeds = clap.encode_text(input_ids, attention_mask=attention_mask, normalize=True)

    # Compute similarity
    similarity = clap.similarity(audio_embeds, text_embeds)  # [1, num_tags]

    # For tagging, use sigmoid to get independent probabilities
    # Scale similarity to reasonable range for sigmoid
    probs = mx.sigmoid(similarity)[0]  # [num_tags]

    # Get active tags
    active_mask = probs >= threshold
    active_indices = mx.where(active_mask)[0]
    active_tags = [tags[int(i)] for i in active_indices]

    return TaggingResult(
        tags=active_tags,
        probabilities=probs,
        tag_names=tags,
        threshold=threshold,
        model_name=model,
        metadata={"sample_rate": sample_rate, "method": "zero_shot"},
    )


def _trained_tag(
    audio: "mx.array",
    sample_rate: int,
    model: str,
    tags: list[str] | None,
    threshold: float,
    **kwargs,
) -> TaggingResult:
    """Perform tagging using a trained tagger."""
    import mlx.core as mx
    from pathlib import Path

    from mlx_audio.models.classifier import CLAPClassifier

    # Load tagger
    if Path(model).exists():
        tagger = CLAPClassifier.from_pretrained(model)
    else:
        from mlx_audio.hub.cache import get_cache
        cache = get_cache()
        tagger = cache.get_model(model, CLAPClassifier)

    # Use tags from tagger if not provided
    tag_names = tags or tagger.label_names

    # Resample if needed
    clap_sr = tagger.clap.config.audio.sample_rate
    if sample_rate != clap_sr:
        from mlx_audio.primitives import resample
        audio = resample(audio, sample_rate, clap_sr)
        sample_rate = clap_sr

    # Get predictions
    logits = tagger(audio)
    probs = mx.sigmoid(logits)[0]  # First sample, sigmoid for multi-label

    # Get active tags
    active_mask = probs >= threshold
    active_indices = mx.where(active_mask)[0]

    if tag_names:
        active_tags = [tag_names[int(i)] for i in active_indices]
    else:
        active_tags = [int(i) for i in active_indices]

    return TaggingResult(
        tags=active_tags,
        probabilities=probs,
        tag_names=tag_names,
        threshold=threshold,
        model_name=model,
        metadata={"sample_rate": sample_rate, "method": "trained"},
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
