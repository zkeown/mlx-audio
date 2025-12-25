"""mlx-audio: Complete audio ML toolkit for Apple Silicon.

A unified library for audio processing, training, and inference using Apple's MLX framework.

Quick Start:
    >>> import mlx_audio
    >>> stems = mlx_audio.separate("song.mp3")
    >>> stems.vocals  # Isolated vocal track
    >>> stems.drums   # Isolated drum track

    >>> result = mlx_audio.embed(audio="dog.wav", text=["dog", "cat", "bird"])
    >>> result.best_match()  # "dog"

    >>> text = mlx_audio.transcribe("speech.wav")  # Coming soon
    >>> print(text.text)

    >>> audio = mlx_audio.generate("jazz piano, upbeat mood")  # Coming soon
    >>> audio.save("output.wav")

Submodules:
    - mlx_audio.primitives: Audio DSP operations (STFT, Mel, MFCC, etc.)
    - mlx_audio.data: DataLoader and dataset utilities
    - mlx_audio.train: Training framework (Trainer, TrainModule, callbacks)
    - mlx_audio.models: Pre-built model implementations (Demucs, CLAP, etc.)
    - mlx_audio.functional: High-level task functions
    - mlx_audio.hub: Model registry and caching
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from mlx_audio._version import __version__

if TYPE_CHECKING:
    import mlx.core as mx
    import numpy as np

    from mlx_audio.types import (
        CLAPEmbeddingResult,
        ClassificationResult,
        DiarizationResult,
        EnhancementResult,
        GenerationResult,
        SeparationResult,
        SpeechResult,
        TaggingResult,
        TranscriptionResult,
        VADResult,
    )

# Type alias for audio input
AudioInput = "str | np.ndarray | mx.array"

# =============================================================================
# HIGH-LEVEL API (One-liner functions)
# =============================================================================


def separate(
    audio: AudioInput,
    *,
    model: str = "htdemucs_ft",
    stems: list[str] | None = None,
    **kwargs: Any,
) -> "SeparationResult":
    """Separate audio into stems (vocals, drums, bass, other).

    Args:
        audio: Path to audio file, numpy array, or MLX array
        model: Model identifier (default: "htdemucs_ft")
        stems: Specific stems to return (None = all available)
        **kwargs: Model-specific parameters (segment, overlap, shifts)

    Returns:
        SeparationResult with stems accessible as attributes

    Example:
        >>> stems = mlx_audio.separate("song.mp3")
        >>> stems.vocals  # Vocal track
        >>> stems.drums   # Drum track
        >>> stems.save("output/")  # Save all stems
    """
    from mlx_audio.functional.separate import separate as _separate

    return _separate(audio, model=model, stems=stems, **kwargs)


def transcribe(
    audio: AudioInput,
    *,
    model: str = "whisper-large-v3-turbo",
    language: str | None = None,
    task: str = "transcribe",
    temperature: float = 0.0,
    beam_size: int = 1,
    word_timestamps: bool = False,
    **kwargs: Any,
) -> "TranscriptionResult":
    """Transcribe speech to text.

    Args:
        audio: Path to audio file or array
        model: Whisper model identifier (e.g., "whisper-large-v3-turbo", "whisper-tiny")
        language: Source language code (None for auto-detect)
        task: "transcribe" or "translate" (translate to English)
        temperature: Sampling temperature (0 = greedy)
        beam_size: Beam search width (1 = greedy)
        word_timestamps: Enable word-level timestamps
        **kwargs: Model-specific parameters

    Returns:
        TranscriptionResult with text and segments

    Example:
        >>> result = mlx_audio.transcribe("speech.wav")
        >>> print(result.text)

        >>> # Save as subtitles
        >>> result.save("subtitles.srt", format="srt")

        >>> # Translate to English
        >>> result = mlx_audio.transcribe("french.wav", task="translate")
    """
    from mlx_audio.functional.transcribe import transcribe as _transcribe

    return _transcribe(
        audio,
        model=model,
        language=language,
        task=task,
        temperature=temperature,
        beam_size=beam_size,
        word_timestamps=word_timestamps,
        **kwargs,
    )


def generate(
    prompt: str,
    *,
    model: str = "musicgen-medium",
    duration: float = 10.0,
    temperature: float = 1.0,
    top_k: int = 250,
    top_p: float = 0.0,
    cfg_scale: float = 3.0,
    seed: int | None = None,
    output_file: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    **kwargs: Any,
) -> "GenerationResult":
    """Generate audio from text description.

    Args:
        prompt: Text description of desired audio (e.g., "jazz piano, upbeat mood")
        model: Generator model identifier (e.g., "musicgen-small", "musicgen-medium")
        duration: Output duration in seconds (max 30s)
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling threshold
        cfg_scale: Classifier-free guidance scale
        seed: Random seed for reproducibility
        output_file: Optional path to save generated audio
        progress_callback: Optional callback for progress updates
        **kwargs: Additional model-specific parameters

    Returns:
        GenerationResult containing generated audio

    Example:
        >>> audio = mlx_audio.generate("jazz piano, upbeat mood")
        >>> audio.play()  # Play the audio
        >>> audio.save("output.wav")  # Save to file
    """
    from mlx_audio.functional.generate import generate as _generate

    return _generate(
        prompt,
        model=model,
        duration=duration,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        cfg_scale=cfg_scale,
        seed=seed,
        output_file=output_file,
        progress_callback=progress_callback,
        **kwargs,
    )


def embed(
    audio: "AudioInput | None" = None,
    text: str | list[str] | None = None,
    *,
    model: str = "clap-htsat-fused",
    return_similarity: bool = False,
    **kwargs: Any,
) -> "CLAPEmbeddingResult":
    """Compute audio and/or text embeddings using CLAP.

    CLAP (Contrastive Language-Audio Pretraining) encodes audio and text
    into a shared embedding space, enabling similarity search, zero-shot
    classification, and audio-text retrieval.

    Args:
        audio: Path to audio file, numpy array, or MLX array
        text: Text string or list of text strings
        model: Embedding model identifier (default: "clap-htsat-fused")
        return_similarity: Compute audio-text similarity matrix
        **kwargs: Model-specific parameters

    Returns:
        CLAPEmbeddingResult with embeddings and optional similarity

    Examples:
        # Audio embedding
        >>> result = mlx_audio.embed(audio="dog_bark.wav")
        >>> result.audio_embeds  # [1, 512]

        # Text embedding
        >>> result = mlx_audio.embed(text="a dog barking loudly")
        >>> result.text_embeds  # [1, 512]

        # Zero-shot classification
        >>> result = mlx_audio.embed(
        ...     audio="sound.wav",
        ...     text=["dog barking", "cat meowing", "bird singing"],
        ...     return_similarity=True,
        ... )
        >>> result.best_match()  # "dog barking"
    """
    from mlx_audio.functional.embed import embed as _embed

    return _embed(
        audio=audio,
        text=text,
        model=model,
        return_similarity=return_similarity,
        **kwargs,
    )


def detect_speech(
    audio: AudioInput,
    *,
    model: str = "silero-vad",
    threshold: float = 0.5,
    min_speech_duration: float = 0.25,
    min_silence_duration: float = 0.1,
    return_probabilities: bool = False,
    **kwargs: Any,
) -> "VADResult":
    """Detect speech segments in audio.

    Voice Activity Detection (VAD) identifies which portions of audio contain
    human speech. Useful for preprocessing before transcription, reducing
    computation, or analyzing audio for speech presence.

    Args:
        audio: Path to audio file, numpy array, or MLX array
        model: VAD model identifier (default: "silero-vad")
        threshold: Speech probability threshold (0.0 to 1.0)
        min_speech_duration: Minimum speech segment duration in seconds
        min_silence_duration: Minimum silence to split segments in seconds
        return_probabilities: Include per-frame probabilities in result
        **kwargs: Model-specific parameters

    Returns:
        VADResult with detected speech segments

    Examples:
        >>> result = mlx_audio.detect_speech("audio.wav")
        >>> for seg in result.segments:
        ...     print(f"Speech: {seg.start:.2f}s - {seg.end:.2f}s")

        >>> # With custom threshold
        >>> result = mlx_audio.detect_speech("audio.wav", threshold=0.7)

        >>> # Extract speech portions
        >>> speech_audio = result.get_speech_audio(audio_array)
    """
    from mlx_audio.functional.detect_speech import detect_speech as _detect_speech

    return _detect_speech(
        audio,
        model=model,
        threshold=threshold,
        min_speech_duration=min_speech_duration,
        min_silence_duration=min_silence_duration,
        return_probabilities=return_probabilities,
        **kwargs,
    )


def enhance(
    audio: AudioInput,
    *,
    model: str = "deepfilternet2",
    method: str = "neural",
    keep_original: bool = False,
    **kwargs: Any,
) -> "EnhancementResult":
    """Enhance audio quality by removing noise and improving clarity.

    Supports both neural (DeepFilterNet) and non-neural (spectral gating)
    enhancement methods. Neural methods provide higher quality but require
    model download; spectral methods work offline with no dependencies.

    Args:
        audio: Path to audio file, numpy array, or MLX array
        model: Enhancement model identifier (default: "deepfilternet2")
        method: "neural" (model-based) or "spectral" (spectral gating)
        keep_original: Store original audio for comparison in result
        **kwargs: Method-specific parameters

    Returns:
        EnhancementResult with enhanced audio

    Examples:
        >>> # Neural enhancement (best quality)
        >>> result = mlx_audio.enhance("noisy_speech.wav")
        >>> result.save("clean.wav")

        >>> # Spectral gating (no model download needed)
        >>> result = mlx_audio.enhance("audio.wav", method="spectral")

        >>> # Compare before/after
        >>> result = mlx_audio.enhance("noisy.wav", keep_original=True)
        >>> original, enhanced = result.before_after
    """
    from mlx_audio.functional.enhance import enhance as _enhance

    return _enhance(
        audio,
        model=model,
        method=method,
        keep_original=keep_original,
        **kwargs,
    )


def speak(
    text: str,
    *,
    model: str = "parler-tts-mini",
    description: str | None = None,
    duration: float | None = None,
    speed: float = 1.0,
    temperature: float = 1.0,
    top_k: int = 50,
    seed: int | None = None,
    output_file: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    **kwargs: Any,
) -> "SpeechResult":
    """Convert text to speech.

    Generate natural speech from text using Parler-TTS. Supports voice
    customization via natural language descriptions.

    Args:
        text: Text to synthesize into speech
        model: TTS model identifier (e.g., "parler-tts-mini", "parler-tts-large")
        description: Voice description for speaker characteristics
                    (e.g., "A warm female voice, speaking clearly")
        duration: Optional target duration in seconds (None = automatic)
        speed: Speech speed multiplier (0.5 = half speed, 2.0 = double speed)
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        seed: Random seed for reproducibility
        output_file: Optional path to save generated audio
        progress_callback: Optional callback for progress updates
        **kwargs: Additional model-specific parameters

    Returns:
        SpeechResult containing generated speech audio

    Examples:
        >>> audio = mlx_audio.speak("Hello, how are you today?")
        >>> audio.play()  # Play the speech
        >>> audio.save("output.wav")  # Save to file

        >>> # With voice description
        >>> audio = mlx_audio.speak(
        ...     "Welcome to the presentation.",
        ...     description="A professional male voice, clear and authoritative",
        ... )

        >>> # Adjust speed
        >>> audio = mlx_audio.speak("Slow speech example", speed=0.8)
    """
    from mlx_audio.functional.speak import speak as _speak

    return _speak(
        text,
        model=model,
        description=description,
        duration=duration,
        speed=speed,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
        output_file=output_file,
        progress_callback=progress_callback,
        **kwargs,
    )


def diarize(
    audio: AudioInput,
    *,
    model: str = "ecapa-tdnn",
    num_speakers: int | None = None,
    min_speakers: int = 1,
    max_speakers: int | None = None,
    return_embeddings: bool = False,
    transcription: "TranscriptionResult | None" = None,
    **kwargs: Any,
) -> "DiarizationResult":
    """Identify who spoke when in audio (speaker diarization).

    Segments audio by speaker identity, answering "who spoke when" without
    necessarily identifying who the speakers are. Can be combined with
    transcription to produce speaker-labeled transcripts.

    Args:
        audio: Path to audio file, numpy array, or MLX array
        model: Diarization model identifier (default: "ecapa-tdnn")
        num_speakers: Known number of speakers (None for auto-detection)
        min_speakers: Minimum speakers for auto-detection
        max_speakers: Maximum speakers for auto-detection
        return_embeddings: Include speaker embeddings in result
        transcription: Optional TranscriptionResult to assign speakers to
        **kwargs: Model-specific parameters

    Returns:
        DiarizationResult with speaker segments

    Examples:
        >>> result = mlx_audio.diarize("meeting.wav")
        >>> for seg in result.segments:
        ...     print(f"{seg.speaker}: {seg.start:.1f}s - {seg.end:.1f}s")

        >>> # With known number of speakers
        >>> result = mlx_audio.diarize("interview.wav", num_speakers=2)

        >>> # Combined with transcription
        >>> transcript = mlx_audio.transcribe("meeting.wav")
        >>> result = mlx_audio.diarize("meeting.wav", transcription=transcript)
        >>> for seg in result.segments:
        ...     print(f"{seg.speaker}: {seg.text}")
    """
    from mlx_audio.functional.diarize import diarize as _diarize

    return _diarize(
        audio,
        model=model,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        return_embeddings=return_embeddings,
        transcription=transcription,
        **kwargs,
    )


def classify(
    audio: AudioInput,
    *,
    model: str = "clap-htsat-fused",
    labels: list[str] | None = None,
    top_k: int = 1,
    **kwargs: Any,
) -> "ClassificationResult":
    """Classify audio into predefined categories.

    Uses CLAP embeddings with zero-shot classification via text similarity.
    For trained classifiers, specify a model path or registry name.

    Args:
        audio: Path to audio file, numpy array, or MLX array
        model: Model name, path, or CLAP model for zero-shot classification
        labels: Class labels for zero-shot classification (required for CLAP)
        top_k: Number of top predictions to return
        **kwargs: Model-specific parameters

    Returns:
        ClassificationResult with prediction and probabilities

    Examples:
        >>> # Zero-shot classification
        >>> result = mlx_audio.classify(
        ...     "sound.wav",
        ...     labels=["dog barking", "cat meowing", "bird singing"]
        ... )
        >>> print(f"Class: {result.predicted_class} ({result.confidence:.1%})")

        >>> # With trained classifier
        >>> result = mlx_audio.classify("audio.wav", model="./my_classifier")
        >>> result.top_k_classes  # ["speech", "music"]
    """
    from mlx_audio.functional.classify import classify as _classify

    return _classify(audio, model=model, labels=labels, top_k=top_k, **kwargs)


def tag(
    audio: AudioInput,
    *,
    model: str = "clap-htsat-fused",
    tags: list[str] | None = None,
    threshold: float = 0.5,
    **kwargs: Any,
) -> "TaggingResult":
    """Tag audio with multiple labels (multi-label classification).

    Uses CLAP embeddings with zero-shot tagging via text similarity.
    For trained taggers, specify a model path or registry name.

    Args:
        audio: Path to audio file, numpy array, or MLX array
        model: Model name, path, or CLAP model for zero-shot tagging
        tags: Tag labels for zero-shot tagging (required for CLAP)
        threshold: Probability threshold for active tags
        **kwargs: Model-specific parameters

    Returns:
        TaggingResult with active tags and probabilities

    Examples:
        >>> # Zero-shot tagging
        >>> result = mlx_audio.tag(
        ...     "music.wav",
        ...     tags=["piano", "guitar", "drums", "vocals", "bass"]
        ... )
        >>> result.tags  # ["piano", "vocals"]
        >>> result.top_k(3)  # [("piano", 0.92), ("vocals", 0.85), ("guitar", 0.45)]

        >>> # With trained tagger
        >>> result = mlx_audio.tag("audio.wav", model="./audioset_tagger")
        >>> for t, prob in result.above_threshold():
        ...     print(f"{t}: {prob:.1%}")
    """
    from mlx_audio.functional.tag import tag as _tag

    return _tag(audio, model=model, tags=tags, threshold=threshold, **kwargs)


# =============================================================================
# LICENSE UTILITIES (Re-exported for convenience)
# =============================================================================

# =============================================================================
# DATA LOADING (Re-exported for convenience)
# =============================================================================
from mlx_audio.data import (
    DataLoader,
    Dataset,
    StreamingDataset,
)
from mlx_audio.hub.licenses import (
    is_commercial_safe,
    list_commercial_safe_models,
    list_non_commercial_models,
)

# =============================================================================
# PRIMITIVES (Re-exported for convenience)
# =============================================================================
from mlx_audio.primitives import (
    griffinlim,
    istft,
    melspectrogram,
    mfcc,
    resample,
    stft,
)

# =============================================================================
# TRAINING (Re-exported for convenience)
# =============================================================================
from mlx_audio.train import (
    OptimizerConfig,
    Trainer,
    TrainModule,
)

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Version
    "__version__",
    # High-level API
    "separate",
    "transcribe",
    "generate",
    "speak",
    "embed",
    "detect_speech",
    "enhance",
    "diarize",
    "classify",
    "tag",
    # License utilities
    "is_commercial_safe",
    "list_commercial_safe_models",
    "list_non_commercial_models",
    # Primitives
    "stft",
    "istft",
    "melspectrogram",
    "mfcc",
    "resample",
    "griffinlim",
    # Data
    "DataLoader",
    "Dataset",
    "StreamingDataset",
    # Train
    "TrainModule",
    "Trainer",
    "OptimizerConfig",
]
