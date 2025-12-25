"""Audio classification API."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx
    import numpy as np

from mlx_audio.constants import CLAP_SAMPLE_RATE
from mlx_audio.functional._audio import ensure_mono_batch, load_audio_input
from mlx_audio.functional._clap_tasks import (
    clap_zero_shot_inference,
    get_top_k_predictions,
)
from mlx_audio.functional._model_utils import is_clap_model
from mlx_audio.types.results import ClassificationResult


def classify(
    audio: str | Path | np.ndarray | mx.array,
    *,
    model: str = "clap-htsat-fused",
    labels: list[str] | None = None,
    top_k: int = 1,
    sample_rate: int | None = None,
    **kwargs,
) -> ClassificationResult:
    """Classify audio into predefined categories.

    Uses CLAP embeddings with zero-shot classification via text similarity.
    For trained classifiers, specify a model path or registry name.

    Args:
        audio: Path to audio file, numpy array [C, T] or [T], or MLX array
        model: Model name, path, or CLAP model for zero-shot classification
        labels: Class labels for zero-shot (required for CLAP models)
        top_k: Number of top predictions to return
        sample_rate: Audio sample rate (inferred from file if not provided)
        **kwargs: Additional model parameters

    Returns:
        ClassificationResult with prediction and probabilities

    Raises:
        AudioLoadError: If audio cannot be loaded or is invalid
        ConfigurationError: If labels are not provided for zero-shot mode
        ModelNotFoundError: If model cannot be found

    Examples:
        Zero-shot classification:
        >>> result = classify(
        ...     "sound.wav",
        ...     labels=["dog barking", "cat meowing", "bird singing"]
        ... )
        >>> print(f"Class: {result.predicted_class} ({result.confidence:.1%})")

        With trained classifier:
        >>> result = classify("audio.wav", model="./my_classifier")
        >>> result.top_k_classes  # ["speech", "music"]
    """
    # Load and preprocess audio using shared utility
    audio_array, sr = load_audio_input(
        audio,
        sample_rate=sample_rate,
        default_sample_rate=CLAP_SAMPLE_RATE,
    )

    # Ensure mono with batch dimension [B, T]
    audio_array = ensure_mono_batch(audio_array)

    # Check if this is a CLAP model (for zero-shot) or trained classifier
    if is_clap_model(model):
        return _zero_shot_classify(
            audio_array, sr, model, labels, top_k, **kwargs
        )
    else:
        return _trained_classify(
            audio_array, sr, model, labels, top_k, **kwargs
        )


def _zero_shot_classify(
    audio: mx.array,
    sample_rate: int,
    model: str,
    labels: list[str] | None,
    top_k: int,
    **kwargs,
) -> ClassificationResult:
    """Perform zero-shot classification using CLAP text-audio similarity."""
    # Use shared CLAP inference logic
    probs, sample_rate = clap_zero_shot_inference(
        audio=audio,
        sample_rate=sample_rate,
        model=model,
        labels=labels,
        task_type="classify",
        **kwargs,
    )

    # Get top-k predictions using shared utility
    top_k_classes, top_k_probs, predicted_class, _ = get_top_k_predictions(
        probs, labels, top_k
    )

    return ClassificationResult(
        predicted_class=predicted_class,
        probabilities=probs,
        class_names=labels,
        top_k_classes=top_k_classes,
        top_k_probs=top_k_probs,
        model_name=model,
        metadata={"sample_rate": sample_rate, "method": "zero_shot"},
    )


def _trained_classify(
    audio: mx.array,
    sample_rate: int,
    model: str,
    labels: list[str] | None,
    top_k: int,
    **kwargs,
) -> ClassificationResult:
    """Perform classification using a trained classifier."""
    from pathlib import Path

    import mlx.core as mx

    from mlx_audio.models.classifier import CLAPClassifier

    # Load classifier
    if Path(model).exists():
        classifier = CLAPClassifier.from_pretrained(model)
    else:
        from mlx_audio.hub.cache import get_cache
        cache = get_cache()
        classifier = cache.get_model(model, CLAPClassifier)

    # Use labels from classifier if not provided
    class_names = labels or classifier.label_names

    # Resample if needed
    clap_sr = classifier.clap.config.audio.sample_rate
    if sample_rate != clap_sr:
        from mlx_audio.primitives import resample
        audio = resample(audio, sample_rate, clap_sr)
        sample_rate = clap_sr

    # Get predictions
    logits = classifier(audio)
    probs = mx.softmax(logits, axis=-1)[0]  # First sample

    # Get top-k predictions
    top_k_classes, top_k_probs, predicted_class, _ = get_top_k_predictions(
        probs, class_names, top_k
    ) if class_names else (None, None, None, None)

    if class_names:
        sorted_indices = mx.argsort(probs)[::-1]
        top_k_indices = sorted_indices[:top_k]
        top_k_probs = [float(probs[i]) for i in top_k_indices]
        top_k_classes = [class_names[int(i)] for i in top_k_indices]
        predicted_class = class_names[int(top_k_indices[0])]
    else:
        sorted_indices = mx.argsort(probs)[::-1]
        top_k_indices = sorted_indices[:top_k]
        top_k_probs = [float(probs[i]) for i in top_k_indices]
        top_k_classes = [int(i) for i in top_k_indices]
        predicted_class = int(top_k_indices[0])

    return ClassificationResult(
        predicted_class=predicted_class,
        probabilities=probs,
        class_names=class_names,
        top_k_classes=top_k_classes,
        top_k_probs=top_k_probs,
        model_name=model,
        metadata={"sample_rate": sample_rate, "method": "trained"},
    )
