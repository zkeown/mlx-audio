"""Audio tagging API."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx
    import numpy as np

from mlx_audio.constants import CLAP_SAMPLE_RATE
from mlx_audio.functional._audio import load_audio_input, ensure_mono_batch
from mlx_audio.functional._clap_tasks import (
    clap_zero_shot_inference,
    get_active_tags,
)
from mlx_audio.functional._model_utils import is_clap_model
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
        tags: Tag labels for zero-shot (required for CLAP models)
        threshold: Probability threshold for active tags
        sample_rate: Audio sample rate (inferred from file if not provided)
        **kwargs: Additional model parameters

    Returns:
        TaggingResult with active tags and probabilities

    Raises:
        AudioLoadError: If audio cannot be loaded or is invalid
        ConfigurationError: If tags are not provided for zero-shot mode
        ModelNotFoundError: If model cannot be found

    Examples:
        Zero-shot tagging:
        >>> result = tag(
        ...     "music.wav",
        ...     tags=["piano", "guitar", "drums", "vocals", "bass"]
        ... )
        >>> result.tags  # ["piano", "vocals"]
        >>> result.top_k(3)  # [("piano", 0.92), ("vocals", 0.85), ...]

        With trained tagger:
        >>> result = tag("audio.wav", model="./audioset_tagger", threshold=0.3)
        >>> for t, prob in result.above_threshold():
        ...     print(f"{t}: {prob:.1%}")
    """
    # Load and preprocess audio using shared utility
    audio_array, sr = load_audio_input(
        audio,
        sample_rate=sample_rate,
        default_sample_rate=CLAP_SAMPLE_RATE,
    )

    # Ensure mono with batch dimension [B, T]
    audio_array = ensure_mono_batch(audio_array)

    # Check if this is a CLAP model (for zero-shot) or trained tagger
    if is_clap_model(model):
        return _zero_shot_tag(
            audio_array, sr, model, tags, threshold, **kwargs
        )
    else:
        return _trained_tag(
            audio_array, sr, model, tags, threshold, **kwargs
        )


def _zero_shot_tag(
    audio: "mx.array",
    sample_rate: int,
    model: str,
    tags: list[str] | None,
    threshold: float,
    **kwargs,
) -> TaggingResult:
    """Perform zero-shot tagging using CLAP text-audio similarity."""
    # Use shared CLAP inference logic
    probs, sample_rate = clap_zero_shot_inference(
        audio=audio,
        sample_rate=sample_rate,
        model=model,
        labels=tags,
        task_type="tag",
        **kwargs,
    )

    # Get active tags using shared utility
    active_tags = get_active_tags(probs, tags, threshold)

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
    if tag_names:
        active_tags = get_active_tags(probs, tag_names, threshold)
    else:
        active_tags = []

    if not tag_names:
        active_mask = probs >= threshold
        active_indices = mx.where(active_mask)[0]
        active_tags = [int(i) for i in active_indices]

    return TaggingResult(
        tags=active_tags,
        probabilities=probs,
        tag_names=tag_names,
        threshold=threshold,
        model_name=model,
        metadata={"sample_rate": sample_rate, "method": "trained"},
    )
