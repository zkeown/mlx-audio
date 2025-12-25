"""Shared CLAP-based inference utilities.

This module provides common functionality for CLAP-based tasks
like zero-shot classification and tagging, eliminating duplication
between classify.py and tag.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import mlx.core as mx

from mlx_audio.exceptions import ConfigurationError


def clap_zero_shot_inference(
    audio: mx.array,
    sample_rate: int,
    model: str,
    labels: list[str],
    task_type: Literal["classify", "tag"],
    **kwargs,
) -> tuple[mx.array, int]:
    """Perform zero-shot inference using CLAP text-audio similarity.

    This function handles the shared logic for both classification (single-label)
    and tagging (multi-label) tasks:
    1. Load CLAP model from cache
    2. Resample audio if needed
    3. Encode audio and text
    4. Compute similarity
    5. Apply appropriate activation (softmax for classify, sigmoid for tag)

    Args:
        audio: Audio array with batch dimension [B, T]
        sample_rate: Audio sample rate
        model: CLAP model name or path
        labels: List of text labels for zero-shot inference
        task_type: "classify" for single-label (softmax) or "tag" for multi-label (sigmoid)
        **kwargs: Additional arguments passed to model

    Returns:
        Tuple of (probabilities, sample_rate) where probabilities shape is [num_labels]

    Raises:
        ConfigurationError: If labels are not provided or empty
    """
    import mlx.core as mx

    if labels is None or len(labels) == 0:
        task_name = "classification" if task_type == "classify" else "tagging"
        example_arg = "labels" if task_type == "classify" else "tags"
        example_values = "['dog', 'cat', 'bird']" if task_type == "classify" else "['piano', 'guitar', 'drums']"
        raise ConfigurationError(
            f"Labels must be provided for zero-shot {task_name}. "
            f"Example: {task_type}(audio, {example_arg}={example_values})"
        )

    from mlx_audio.hub.cache import get_cache
    from mlx_audio.models.clap import CLAP

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

    # Tokenize and encode labels
    from mlx_audio.functional.embed import _tokenize_text
    input_ids, attention_mask = _tokenize_text(labels)
    text_embeds = clap.encode_text(input_ids, attention_mask=attention_mask, normalize=True)

    # Compute similarity
    similarity = clap.similarity(audio_embeds, text_embeds)  # [1, num_labels]

    # Apply appropriate activation
    if task_type == "classify":
        # Single-label: softmax for mutually exclusive probabilities
        probs = mx.softmax(similarity, axis=-1)[0]  # [num_labels]
    else:
        # Multi-label: sigmoid for independent probabilities
        probs = mx.sigmoid(similarity)[0]  # [num_labels]

    return probs, sample_rate


def get_top_k_predictions(
    probs: mx.array,
    labels: list[str],
    top_k: int,
) -> tuple[list[str], list[float], str, int]:
    """Extract top-k predictions from probability distribution.

    Args:
        probs: Probability array [num_labels]
        labels: List of label names
        top_k: Number of top predictions to return

    Returns:
        Tuple of (top_k_classes, top_k_probs, predicted_class, predicted_idx)
    """
    import mlx.core as mx

    sorted_indices = mx.argsort(probs)[::-1]
    top_k_indices = sorted_indices[:top_k]
    top_k_probs = [float(probs[i]) for i in top_k_indices]
    top_k_classes = [labels[int(i)] for i in top_k_indices]

    predicted_idx = int(top_k_indices[0])
    predicted_class = labels[predicted_idx]

    return top_k_classes, top_k_probs, predicted_class, predicted_idx


def get_active_tags(
    probs: mx.array,
    tags: list[str],
    threshold: float,
) -> list[str]:
    """Get tags with probability above threshold.

    Args:
        probs: Probability array [num_tags]
        tags: List of tag names
        threshold: Probability threshold for active tags

    Returns:
        List of active tag names
    """
    import mlx.core as mx

    active_mask = probs >= threshold
    active_indices = mx.where(active_mask)[0]
    return [tags[int(i)] for i in active_indices]
