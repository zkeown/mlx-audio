"""Audio source separation API."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import mlx.core as mx
    import numpy as np

    from mlx_audio.types.results import SeparationResult


def separate(
    audio: str | Path | "np.ndarray" | "mx.array",
    *,
    model: str = "htdemucs_ft",
    ensemble: bool = False,
    stems: list[str] | None = None,
    output_dir: str | Path | None = None,
    sample_rate: int | None = None,
    segment: float = 6.0,
    overlap: float = 0.25,
    device: str | None = None,
    progress_callback: Any = None,
    **kwargs,
) -> "SeparationResult":
    """Separate audio into stems.

    This is the main entry point for audio source separation.

    Args:
        audio: Path to audio file or audio array [C, T]
        model: Model name or path ("htdemucs_ft", "htdemucs_6s", or path)
        ensemble: Use ensemble of 4 specialized models (~3dB SDR improvement)
        stems: Which sources to return (None = all)
        output_dir: Optional directory to save separated stems
        sample_rate: Sample rate (inferred from file if not provided)
        segment: Segment duration for chunked processing (seconds)
        overlap: Overlap ratio between segments
        device: "gpu" or "cpu"
        progress_callback: Optional progress callback
        **kwargs: Additional model-specific parameters

    Returns:
        SeparationResult with stems accessible as attributes

    Example:
        >>> stems = mlx_audio.separate("song.mp3")
        >>> stems.vocals  # Vocal track
        >>> stems.drums   # Drum track

        >>> # Use ensemble for better quality (~3dB SDR improvement)
        >>> stems = mlx_audio.separate("song.mp3", ensemble=True)

        >>> # Save to files
        >>> mlx_audio.separate("song.mp3", output_dir="./stems")

        >>> # Select specific stems
        >>> stems = mlx_audio.separate(
        ...     "song.mp3",
        ...     model="htdemucs_6s",
        ...     stems=["drums", "bass", "guitar"]
        ... )
    """
    # Import here to avoid circular imports and allow lazy loading
    from mlx_audio.models.demucs import HTDemucs, BagOfModels, apply_model
    from mlx_audio.hub.cache import get_cache
    from mlx_audio.types.results import SeparationResult, AudioData
    from mlx_audio.functional._audio import load_audio_input

    # Load audio using shared utility (44100 Hz default for HTDemucs)
    audio_array, sample_rate = load_audio_input(
        audio,
        sample_rate=sample_rate,
        default_sample_rate=44100,
    )

    # Load model (with caching)
    if ensemble:
        from mlx_audio.exceptions import ModelNotFoundError

        # Load BagOfModels ensemble (4 specialized models)
        cache_dir = Path.home() / ".cache/mlx_audio/models"
        bag_path = cache_dir / f"{model}_bag"
        if not bag_path.exists():
            raise ModelNotFoundError(
                f"Ensemble model not found at {bag_path}. "
                "Please ensure the htdemucs_ft_bag models are downloaded."
            )
        htdemucs = BagOfModels.from_pretrained(bag_path)
        htdemucs.eval()
    else:
        cache = get_cache()
        htdemucs = cache.get_model(model, HTDemucs)

    # Resample if needed
    if sample_rate != htdemucs.config.samplerate:
        from mlx_audio.primitives import resample

        audio_array = resample(audio_array, sample_rate, htdemucs.config.samplerate)

    # Run separation
    stems_array = apply_model(
        htdemucs,
        audio_array,
        segment=segment,
        overlap=overlap,
        progress_callback=progress_callback,
        **kwargs,
    )

    # Build result dictionary
    all_sources = htdemucs.config.sources
    if stems is None:
        stems = all_sources

    result_stems = {}
    for i, source_name in enumerate(all_sources):
        if source_name in stems:
            result_stems[source_name] = AudioData(
                array=stems_array[i],
                sample_rate=htdemucs.config.samplerate,
            )

    result = SeparationResult(
        stems=result_stems,
        sample_rate=htdemucs.config.samplerate,
        model_name=model,
    )

    # Save to files if output_dir provided
    if output_dir:
        result.save(output_dir)

    return result
