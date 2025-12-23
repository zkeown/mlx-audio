"""High-level speaker diarization API."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from mlx_audio.types.results import DiarizationResult, TranscriptionResult


def diarize(
    audio: str | Path | np.ndarray | mx.array,
    *,
    model: str = "ecapa-tdnn",
    num_speakers: int | None = None,
    min_speakers: int = 1,
    max_speakers: int | None = None,
    sample_rate: int | None = None,
    return_embeddings: bool = False,
    transcription: "TranscriptionResult | None" = None,
    **kwargs,
) -> "DiarizationResult":
    """Identify who spoke when in audio.

    Speaker diarization segments audio by speaker identity, answering
    "who spoke when" without necessarily identifying who the speakers are.

    Parameters
    ----------
    audio : str, Path, ndarray, or mx.array
        Path to audio file or audio array.
    model : str, default="ecapa-tdnn"
        Model identifier for speaker embeddings.
    num_speakers : int, optional
        Number of speakers. If None, auto-detect.
    min_speakers : int, default=1
        Minimum number of speakers (for auto-detection).
    max_speakers : int, optional
        Maximum number of speakers (for auto-detection).
    sample_rate : int, optional
        Audio sample rate. Inferred from file if not provided.
    return_embeddings : bool, default=False
        Include speaker embeddings in result.
    transcription : TranscriptionResult, optional
        If provided, assigns speaker labels to transcription segments.
    **kwargs
        Additional model-specific parameters.

    Returns
    -------
    DiarizationResult
        Diarization result with speaker segments.

    Examples
    --------
    >>> result = mlx_audio.diarize("meeting.wav")
    >>> for seg in result.segments:
    ...     print(f"{seg.speaker}: {seg.start:.1f}s - {seg.end:.1f}s")

    >>> # With fixed number of speakers
    >>> result = mlx_audio.diarize("interview.wav", num_speakers=2)

    >>> # Combined with transcription
    >>> transcript = mlx_audio.transcribe("meeting.wav")
    >>> result = mlx_audio.diarize("meeting.wav", transcription=transcript)
    >>> for seg in result.segments:
    ...     print(f"{seg.speaker}: {seg.text}")
    """
    from mlx_audio.types.results import DiarizationResult, SpeakerSegment

    # Load audio if path provided
    if isinstance(audio, (str, Path)):
        audio_array, sr = _load_audio(audio)
        if sample_rate is None:
            sample_rate = sr
    elif isinstance(audio, np.ndarray):
        audio_array = mx.array(audio.astype(np.float32))
        if sample_rate is None:
            sample_rate = 16000  # Default for speech
    else:
        audio_array = audio
        if sample_rate is None:
            sample_rate = 16000

    # Load diarization model
    from mlx_audio.models.diarization import DiarizationConfig, SpeakerDiarization

    config = DiarizationConfig()
    config.sample_rate = sample_rate
    config.min_speakers = min_speakers
    if max_speakers is not None:
        config.max_speakers = max_speakers

    diarizer = SpeakerDiarization(config)

    # Resample if needed
    if sample_rate != config.sample_rate:
        from mlx_audio.primitives import resample
        audio_array = resample(audio_array, sample_rate, config.sample_rate)

    # Run diarization
    raw_segments = diarizer(audio_array, num_speakers=num_speakers)

    # Build SpeakerSegment objects
    segments = []
    for speaker_id, start, end in raw_segments:
        segments.append(SpeakerSegment(
            speaker=speaker_id,
            start=start,
            end=end,
        ))

    # Get embeddings if requested
    speaker_embeddings = None
    if return_embeddings:
        embeddings, _ = diarizer.extract_embeddings(audio_array)
        # Group by speaker
        labels = diarizer.cluster_embeddings(embeddings, num_speakers)
        unique_speakers = np.unique(labels)
        embeddings_np = np.array(embeddings)
        speaker_embeddings = {}
        for spk in unique_speakers:
            mask = labels == spk
            spk_embs = embeddings_np[mask]
            # Average embedding for speaker
            speaker_embeddings[f"SPEAKER_{spk:02d}"] = mx.array(np.mean(spk_embs, axis=0))

    # Merge with transcription if provided
    if transcription is not None:
        segments = _assign_speakers_to_transcript(segments, transcription)

    # Count unique speakers
    unique_speakers_set = set(seg.speaker for seg in segments)

    return DiarizationResult(
        segments=segments,
        num_speakers=len(unique_speakers_set),
        speaker_embeddings=speaker_embeddings,
        model_name=model,
    )


def _load_audio(path: str | Path) -> tuple[mx.array, int]:
    """Load audio file."""
    import soundfile as sf

    path = Path(path)
    audio, sr = sf.read(str(path), dtype="float32")

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return mx.array(audio), sr


def _assign_speakers_to_transcript(
    diarization_segments: list,
    transcription: "TranscriptionResult",
) -> list:
    """Assign speaker labels to transcription segments.

    Uses overlap calculation to determine which speaker corresponds
    to each transcription segment.
    """
    from mlx_audio.types.results import SpeakerSegment

    result_segments = []

    for trans_seg in transcription.segments:
        # Find best matching speaker based on overlap
        best_speaker = None
        best_overlap = 0.0

        for dia_seg in diarization_segments:
            # Calculate overlap
            overlap_start = max(trans_seg.start, dia_seg.start)
            overlap_end = min(trans_seg.end, dia_seg.end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = dia_seg.speaker

        if best_speaker is None:
            best_speaker = "SPEAKER_00"  # Default

        result_segments.append(SpeakerSegment(
            speaker=best_speaker,
            start=trans_seg.start,
            end=trans_seg.end,
            text=trans_seg.text,
            confidence=trans_seg.confidence,
        ))

    return result_segments
