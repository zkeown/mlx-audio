"""Voice Activity Detection API."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx
    import numpy as np

    from mlx_audio.types.vad import VADResult


def detect_speech(
    audio: str | Path | np.ndarray | mx.array,
    *,
    model: str = "silero-vad",
    threshold: float = 0.5,
    min_speech_duration: float = 0.25,
    min_silence_duration: float = 0.1,
    sample_rate: int | None = None,
    return_probabilities: bool = False,
    output_file: str | Path | None = None,
    output_format: str = "json",
    progress_callback: Callable[[float], None] | None = None,
    **kwargs,
) -> VADResult:
    """Detect speech segments in audio.

    This is the main entry point for voice activity detection.

    Args:
        audio: Path to audio file or audio array [T] or [C, T]
        model: Model name (default: "silero-vad")
        threshold: Speech probability threshold (0.0 to 1.0)
        min_speech_duration: Minimum speech segment duration in seconds
        min_silence_duration: Minimum silence duration to split segments in seconds
        sample_rate: Sample rate of input audio (inferred from file if not provided)
        return_probabilities: Include per-frame probabilities in result
        output_file: Optional path to save VAD result
        output_format: Format for output file ("json", "txt", "audacity")
        progress_callback: Optional callback for progress updates [0, 1]
        **kwargs: Additional model-specific parameters

    Returns:
        VADResult with detected speech segments and optional probabilities

    Example:
        >>> # Simple usage
        >>> result = detect_speech("speech.wav")
        >>> for seg in result.segments:
        ...     print(f"Speech: {seg.start:.2f}s - {seg.end:.2f}s")

        >>> # With custom threshold
        >>> result = detect_speech("audio.wav", threshold=0.7)

        >>> # Get per-frame probabilities
        >>> result = detect_speech("audio.wav", return_probabilities=True)
        >>> print(f"Probabilities shape: {result.probabilities.shape}")

        >>> # Extract speech portions
        >>> import mlx.core as mx
        >>> audio = mx.random.normal((16000,))  # 1 second at 16kHz
        >>> result = detect_speech(audio, sample_rate=16000)
        >>> speech_audio = result.get_speech_audio(audio)

        >>> # Save results
        >>> detect_speech("audio.wav", output_file="vad_result.json")
    """
    # Import here to avoid circular imports and allow lazy loading
    import mlx.core as mx

    from mlx_audio.functional._audio import load_audio_input
    from mlx_audio.models.vad import SileroVAD, VADConfig
    from mlx_audio.types.vad import VADResult

    # Load audio using shared utility (16kHz default for VAD, mono)
    audio_array, sample_rate = load_audio_input(
        audio,
        sample_rate=sample_rate,
        default_sample_rate=16000,
        mono=True,
    )

    # Create model config based on sample rate
    config = VADConfig.silero_vad_8k() if sample_rate == 8000 else VADConfig.silero_vad_16k()

    # Override threshold in config
    config.threshold = threshold

    # Create model
    vad_model = SileroVAD(config)

    # Resample if needed
    target_sr = config.sample_rate
    if sample_rate != target_sr:
        from mlx_audio.primitives import resample

        audio_array = resample(audio_array, sample_rate, target_sr)
        sample_rate = target_sr

    # Get window size and calculate number of windows
    window_size = config.window_size_samples
    num_samples = audio_array.shape[0]
    num_windows = (num_samples + window_size - 1) // window_size

    # Pad audio to fit exact windows
    padded_length = num_windows * window_size
    if num_samples < padded_length:
        padding = padded_length - num_samples
        audio_array = mx.pad(audio_array, [(0, padding)])

    # Process each window
    probs = []
    state = None

    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        window = audio_array[start:end]

        prob, state = vad_model(window, state=state)
        probs.append(float(prob))

        # Report progress
        if progress_callback:
            progress_callback((i + 1) / num_windows)

    # Create result from probabilities
    probs_array = mx.array(probs)

    result = VADResult.from_probabilities(
        probabilities=probs_array,
        sample_rate=sample_rate,
        window_size_samples=window_size,
        threshold=threshold,
        min_speech_duration=min_speech_duration,
        min_silence_duration=min_silence_duration,
        model_name=model,
    )

    # Include probabilities if requested
    if not return_probabilities:
        result.probabilities = None

    # Save to file if requested
    if output_file:
        output_file = Path(output_file)
        result.save(output_file, format=output_format)

    return result


__all__ = ["detect_speech"]
