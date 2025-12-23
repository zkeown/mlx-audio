"""Speech transcription API."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import mlx.core as mx
    import numpy as np

    from mlx_audio.types.results import TranscriptionResult


def transcribe(
    audio: str | Path | "np.ndarray" | "mx.array",
    *,
    model: str = "whisper-large-v3-turbo",
    language: str | None = None,
    task: str = "transcribe",
    temperature: float = 0.0,
    beam_size: int = 1,
    word_timestamps: bool = False,
    sample_rate: int | None = None,
    output_file: str | Path | None = None,
    output_format: str = "txt",
    progress_callback: Callable[[float], None] | None = None,
    **kwargs,
) -> "TranscriptionResult":
    """Transcribe speech to text.

    This is the main entry point for speech-to-text transcription.

    Args:
        audio: Path to audio file or audio array [T] or [C, T]
        model: Model name or path (e.g., "whisper-large-v3-turbo", "whisper-tiny")
        language: Language code (e.g., "en", "zh"). None for auto-detect.
        task: "transcribe" or "translate" (translate to English)
        temperature: Sampling temperature (0 = greedy)
        beam_size: Beam search width (1 = greedy)
        word_timestamps: Enable word-level timestamps
        sample_rate: Sample rate of input audio (inferred from file if not provided)
        output_file: Optional path to save transcription
        output_format: Format for output file ("txt", "srt", "vtt", "json")
        progress_callback: Optional callback for progress updates [0, 1]
        **kwargs: Additional model-specific parameters

    Returns:
        TranscriptionResult with text, segments, and timing information

    Example:
        >>> result = mlx_audio.transcribe("speech.wav")
        >>> print(result.text)
        "Hello, how are you today?"

        >>> # With specific language
        >>> result = mlx_audio.transcribe("speech.wav", language="en")

        >>> # Translation to English
        >>> result = mlx_audio.transcribe("french.wav", task="translate")

        >>> # Save as subtitles
        >>> mlx_audio.transcribe("video.mp3", output_file="subs.srt", output_format="srt")

        >>> # With word timestamps
        >>> result = mlx_audio.transcribe("speech.wav", word_timestamps=True)
        >>> for seg in result.segments:
        ...     print(f"[{seg.start:.2f} - {seg.end:.2f}] {seg.text}")
    """
    # Import here to avoid circular imports and allow lazy loading
    from mlx_audio.models.whisper import Whisper, WhisperTokenizer, apply_model
    from mlx_audio.hub.cache import get_cache
    from mlx_audio.types.audio import load_audio
    import mlx.core as mx

    # Load audio if path provided
    if isinstance(audio, (str, Path)):
        audio_array, sr = load_audio(audio, mono=True)
        if sample_rate is None:
            sample_rate = sr
    else:
        import numpy as np

        if isinstance(audio, np.ndarray):
            audio_array = mx.array(audio)
        else:
            audio_array = audio
        if sample_rate is None:
            sample_rate = 16000  # Whisper default

    # Ensure mono audio
    if audio_array.ndim == 2:
        # Take first channel or average
        if audio_array.shape[0] <= 2:
            audio_array = mx.mean(audio_array, axis=0)
        else:
            audio_array = mx.mean(audio_array, axis=1)

    # Load model (with caching)
    cache = get_cache()
    whisper = cache.get_model(model, Whisper)

    # Resample to 16kHz if needed
    target_sr = whisper.config.sample_rate
    if sample_rate != target_sr:
        from mlx_audio.primitives import resample

        audio_array = resample(audio_array, sample_rate, target_sr)

    # Create tokenizer
    tokenizer = WhisperTokenizer(
        multilingual=whisper.config.is_multilingual,
        language=language,
        task=task,
    )

    # Run transcription
    result = apply_model(
        whisper,
        audio_array,
        tokenizer=tokenizer,
        language=language,
        task=task,
        temperature=temperature,
        beam_size=beam_size,
        word_timestamps=word_timestamps,
        progress_callback=progress_callback,
        **kwargs,
    )

    # Update model name in result
    result.model_name = model

    # Save to file if requested
    if output_file:
        output_file = Path(output_file)
        result.save(output_file, format=output_format)

    return result
