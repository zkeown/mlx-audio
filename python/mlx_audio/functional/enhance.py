"""High-level audio enhancement API."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from mlx_audio.types.results import EnhancementResult

from mlx_audio.functional._audio import load_audio_input


def enhance(
    audio: str | Path | np.ndarray | mx.array,
    *,
    model: str = "deepfilternet2",
    sample_rate: int | None = None,
    output_file: str | Path | None = None,
    method: str = "auto",
    keep_original: bool = False,
    **kwargs,
) -> EnhancementResult:
    """Enhance audio quality by removing noise and artifacts.

    Supports both neural enhancement (DeepFilterNet) and simple spectral
    gating for noise reduction.

    Parameters
    ----------
    audio : str, Path, ndarray, or mx.array
        Path to audio file or audio array.
    model : str, default="deepfilternet2"
        Model identifier:
        - "deepfilternet2": Neural enhancement (48kHz)
        - "deepfilternet2-16k": Neural enhancement (16kHz)
        - "spectral": Simple spectral gating (no model)
    sample_rate : int, optional
        Sample rate. Inferred from file if not provided.
    output_file : str or Path, optional
        Path to save enhanced audio.
    method : str, default="auto"
        Enhancement method:
        - "auto": Use model for neural, fallback to spectral
        - "neural": Force neural enhancement
        - "spectral": Force spectral gating
    keep_original : bool, default=False
        If True, store original audio in result metadata.
    **kwargs
        Additional arguments passed to enhancement method:
        - For spectral: threshold_db, prop_decrease, etc.
        - For neural: model-specific parameters

    Returns
    -------
    EnhancementResult
        Enhanced audio with metadata.

    Examples
    --------
    >>> # Basic enhancement with DeepFilterNet
    >>> enhanced = mlx_audio.enhance("noisy_speech.wav")
    >>> enhanced.save("clean_speech.wav")

    >>> # Simple spectral gating (no model download)
    >>> enhanced = mlx_audio.enhance("audio.wav", method="spectral")

    >>> # With custom parameters
    >>> enhanced = mlx_audio.enhance(
    ...     "audio.wav",
    ...     method="spectral",
    ...     threshold_db=-25,
    ...     prop_decrease=0.9,
    ... )

    >>> # Keep original for comparison
    >>> enhanced = mlx_audio.enhance("audio.wav", keep_original=True)
    >>> original, clean = enhanced.before_after
    """
    from mlx_audio.types.results import EnhancementResult

    # Load audio using shared utility
    audio_array, sr = load_audio_input(
        audio,
        sample_rate=sample_rate,
        default_sample_rate=22050,
        mono=True,  # Enhancement typically works on mono
    )
    sample_rate = sr

    original_audio = audio_array if keep_original else None

    # Determine method
    if method == "auto":
        use_neural = model not in ("spectral", "spectral_gate")
    elif method == "neural":
        use_neural = True
    elif method == "spectral":
        use_neural = False
    else:
        from mlx_audio.exceptions import ConfigurationError
        raise ConfigurationError(f"Unknown method: '{method}'")

    # Apply enhancement
    if use_neural:
        enhanced_audio = _enhance_neural(
            audio_array, model, sample_rate, **kwargs
        )
        model_name = model
    else:
        enhanced_audio = _enhance_spectral(
            audio_array, sample_rate, **kwargs
        )
        model_name = "spectral_gate"

    # Build result
    result = EnhancementResult(
        array=enhanced_audio,
        sample_rate=sample_rate,
        model_name=model_name,
        metadata={"original": original_audio} if keep_original else {},
    )

    # Save if requested
    if output_file is not None:
        result.save(output_file)

    return result


def _enhance_neural(
    audio: mx.array,
    model_name: str,
    sample_rate: int,
    **kwargs,
) -> mx.array:
    """Apply neural enhancement."""
    from mlx_audio.models.enhance import DeepFilterNet, DeepFilterNetConfig

    # Select config based on model name
    if model_name == "deepfilternet2-16k":
        config = DeepFilterNetConfig.deepfilternet2_16k()
    else:
        config = DeepFilterNetConfig.deepfilternet2()

    # Resample if needed
    if sample_rate != config.sample_rate:
        from mlx_audio.primitives import resample
        audio = resample(audio, sample_rate, config.sample_rate)
        original_sr = sample_rate
        sample_rate = config.sample_rate
    else:
        original_sr = None

    # Load model (random init for now - would use pretrained in production)
    model = DeepFilterNet(config)

    # Enhance
    enhanced = model(audio)

    # Resample back if needed
    if original_sr is not None:
        from mlx_audio.primitives import resample
        enhanced = resample(enhanced, sample_rate, original_sr)

    return enhanced


def _enhance_spectral(
    audio: mx.array,
    sample_rate: int,
    **kwargs,
) -> mx.array:
    """Apply spectral gating enhancement."""
    from mlx_audio.primitives.spectral_gate import spectral_gate

    # Extract spectral gate parameters
    sg_kwargs = {
        "sr": sample_rate,
        "n_fft": kwargs.get("n_fft", 2048),
        "threshold_db": kwargs.get("threshold_db", -20.0),
        "prop_decrease": kwargs.get("prop_decrease", 1.0),
        "stationary": kwargs.get("stationary", True),
    }

    # Add noise profile if provided
    if "noise_profile" in kwargs:
        sg_kwargs["noise_profile"] = kwargs["noise_profile"]

    return spectral_gate(audio, **sg_kwargs)
