"""Audio classification API."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx
    import numpy as np

from mlx_audio.types.results import ClassificationResult


def classify(
    audio: str | Path | "np.ndarray" | "mx.array",
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
        labels: Class labels for zero-shot classification (required for CLAP models)
        top_k: Number of top predictions to return
        sample_rate: Audio sample rate (inferred from file if not provided)
        **kwargs: Additional model parameters

    Returns:
        ClassificationResult with prediction and probabilities

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

    # Check if this is a CLAP model (for zero-shot) or trained classifier
    if _is_clap_model(model):
        return _zero_shot_classify(
            audio_array, sr, model, labels, top_k, **kwargs
        )
    else:
        return _trained_classify(
            audio_array, sr, model, labels, top_k, **kwargs
        )


def _is_clap_model(model: str) -> bool:
    """Check if model name refers to a CLAP model."""
    clap_models = {"clap-htsat-fused", "clap-htsat-unfused"}
    return model in clap_models or "clap" in model.lower()


def _zero_shot_classify(
    audio: "mx.array",
    sample_rate: int,
    model: str,
    labels: list[str] | None,
    top_k: int,
    **kwargs,
) -> ClassificationResult:
    """Perform zero-shot classification using CLAP text-audio similarity."""
    import mlx.core as mx

    if labels is None or len(labels) == 0:
        raise ValueError(
            "labels must be provided for zero-shot classification. "
            "Example: classify(audio, labels=['dog', 'cat', 'bird'])"
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

    # Tokenize and encode labels
    from mlx_audio.functional.embed import _tokenize_text
    input_ids, attention_mask = _tokenize_text(labels)
    text_embeds = clap.encode_text(input_ids, attention_mask=attention_mask, normalize=True)

    # Compute similarity
    similarity = clap.similarity(audio_embeds, text_embeds)  # [1, num_labels]
    probs = mx.softmax(similarity, axis=-1)[0]  # [num_labels]

    # Get top-k predictions
    sorted_indices = mx.argsort(probs)[::-1]
    top_k_indices = sorted_indices[:top_k]
    top_k_probs = [float(probs[i]) for i in top_k_indices]
    top_k_classes = [labels[int(i)] for i in top_k_indices]

    predicted_idx = int(top_k_indices[0])
    predicted_class = labels[predicted_idx]

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
    audio: "mx.array",
    sample_rate: int,
    model: str,
    labels: list[str] | None,
    top_k: int,
    **kwargs,
) -> ClassificationResult:
    """Perform classification using a trained classifier."""
    import mlx.core as mx
    from pathlib import Path

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
    sorted_indices = mx.argsort(probs)[::-1]
    top_k_indices = sorted_indices[:top_k]
    top_k_probs = [float(probs[i]) for i in top_k_indices]

    if class_names:
        top_k_classes = [class_names[int(i)] for i in top_k_indices]
        predicted_class = class_names[int(top_k_indices[0])]
    else:
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
