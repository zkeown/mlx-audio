"""Result types for mlx-audio tasks.

This module provides structured result types for all mlx-audio operations.
Each result type includes methods for serialization, export, and analysis.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import mlx.core as mx
    import numpy as np

from mlx_audio.types.audio import AudioData


class Result(ABC):
    """Abstract base class for all result types.

    Provides a common interface for serialization and export.
    Subclasses must implement to_dict() and save().

    Example:
        >>> result = mlx_audio.transcribe("speech.wav")
        >>> result.to_dict()  # Get as dictionary
        >>> result.save("output.txt")  # Save to file
        >>> result.to_json("output.json")  # Save as JSON
    """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Convert result to a dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        ...

    @abstractmethod
    def save(self, path: str | Path, **kwargs: Any) -> Path:
        """Save result to a file.

        Args:
            path: Output file path
            **kwargs: Format-specific options

        Returns:
            Path to saved file
        """
        ...

    def to_json(self, path: str | Path, indent: int = 2) -> Path:
        """Save result as JSON file.

        Args:
            path: Output file path
            indent: JSON indentation level

        Returns:
            Path to saved file
        """
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent, default=str)
        return path

    def __repr__(self) -> str:
        """Default representation."""
        class_name = self.__class__.__name__
        return f"{class_name}(...)"


@dataclass
class SeparationResult:
    """Result from audio source separation.

    Provides attribute access to stems and batch operations.

    Attributes:
        stems: Dictionary mapping stem names to AudioData
        sample_rate: Sample rate of all stems
        model_name: Name of the model used
        metadata: Additional metadata from separation
    """

    stems: dict[str, AudioData]
    sample_rate: int
    model_name: str = ""
    metadata: dict = field(default_factory=dict)

    def __getattr__(self, name: str) -> AudioData:
        """Access stems as attributes (e.g., result.vocals)."""
        if name.startswith("_") or name in ("stems", "sample_rate", "model_name", "metadata"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        if name in self.stems:
            return self.stems[name]
        raise AttributeError(
            f"No stem named '{name}'. Available: {list(self.stems.keys())}"
        )

    def save(
        self,
        directory: str | Path,
        *,
        format: str = "wav",
        stem_names: list[str] | None = None,
    ) -> dict[str, Path]:
        """Save all stems to directory.

        Args:
            directory: Output directory
            format: Audio format (wav, flac, mp3)
            stem_names: Specific stems to save (None = all)

        Returns:
            Dict mapping stem names to saved file paths
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        saved = {}
        for name, audio in self.stems.items():
            if stem_names is None or name in stem_names:
                path = directory / f"{name}.{format}"
                audio.save(path)
                saved[name] = path

        return saved

    @property
    def available_stems(self) -> list[str]:
        """List of available stem names."""
        return list(self.stems.keys())

    def __repr__(self) -> str:
        stems_str = ", ".join(self.available_stems)
        return f"SeparationResult(stems=[{stems_str}], sample_rate={self.sample_rate})"


@dataclass
class TranscriptionSegment:
    """A segment of transcribed text with timing."""

    text: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    confidence: float = 0.0


@dataclass
class TranscriptionResult:
    """Result from speech transcription.

    Attributes:
        text: Full transcription text
        segments: List of timed segments
        language: Detected language code
        language_probability: Confidence in language detection
        model_name: Name of the model used
    """

    text: str
    segments: list[TranscriptionSegment] = field(default_factory=list)
    language: str | None = None
    language_probability: float = 0.0
    model_name: str = ""
    metadata: dict = field(default_factory=dict)

    def to_srt(self) -> str:
        """Export as SRT subtitle format."""
        lines = []
        for i, seg in enumerate(self.segments, 1):
            start = self._format_timestamp(seg.start)
            end = self._format_timestamp(seg.end)
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(seg.text.strip())
            lines.append("")
        return "\n".join(lines)

    def to_vtt(self) -> str:
        """Export as WebVTT subtitle format."""
        lines = ["WEBVTT", ""]
        for seg in self.segments:
            start = self._format_timestamp(seg.start, vtt=True)
            end = self._format_timestamp(seg.end, vtt=True)
            lines.append(f"{start} --> {end}")
            lines.append(seg.text.strip())
            lines.append("")
        return "\n".join(lines)

    def save(self, path: str | Path, format: str = "txt") -> Path:
        """Save transcription to file.

        Args:
            path: Output file path
            format: Output format (txt, srt, vtt, json)

        Returns:
            Path to saved file
        """
        import json

        path = Path(path)
        if format == "txt":
            path.write_text(self.text)
        elif format == "srt":
            path.write_text(self.to_srt())
        elif format == "vtt":
            path.write_text(self.to_vtt())
        elif format == "json":
            data = {
                "text": self.text,
                "language": self.language,
                "segments": [
                    {"text": s.text, "start": s.start, "end": s.end}
                    for s in self.segments
                ],
            }
            path.write_text(json.dumps(data, indent=2))
        return path

    def _format_timestamp(self, seconds: float, vtt: bool = False) -> str:
        """Format timestamp for subtitle files."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        sep = "." if vtt else ","
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", sep)


@dataclass
class GenerationResult(AudioData):
    """Result from audio generation.

    Extends AudioData with generation-specific metadata.

    Attributes:
        array: Generated audio data
        sample_rate: Sample rate
        prompt: Text prompt used for generation
        model_name: Name of the model used
        generation_params: Parameters used for generation
    """

    prompt: str = ""
    model_name: str = ""
    generation_params: dict = field(default_factory=dict)

    def play(self) -> None:
        """Play the generated audio (requires sounddevice)."""
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "sounddevice is required for playback. "
                "Install with: pip install sounddevice"
            )
        sd.play(self.to_numpy().T, self.sample_rate)
        sd.wait()


@dataclass
class EmbeddingResult:
    """Result from audio embedding.

    Supports both single and batch embeddings.

    Attributes:
        vectors: Embedding vectors [embedding_dim] or [batch, embedding_dim]
        model_name: Name of the model used
        metadata: Additional metadata
    """

    vectors: mx.array
    model_name: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def vector(self) -> mx.array:
        """Get single embedding vector (first if batched)."""
        if len(self.vectors.shape) == 1:
            return self.vectors
        return self.vectors[0]

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self.vectors.shape[-1]

    def to_numpy(self):
        """Convert to NumPy array."""
        import numpy as np

        return np.array(self.vectors)

    def cosine_similarity(self, other: EmbeddingResult) -> float:
        """Compute cosine similarity with another embedding."""
        import mlx.core as mx

        a = self.vector / mx.linalg.norm(self.vector)
        b = other.vector / mx.linalg.norm(other.vector)
        return float(mx.sum(a * b))


@dataclass
class SpeakerSegment:
    """A segment of speech with speaker assignment."""

    speaker: str  # Speaker ID (e.g., "SPEAKER_00")
    start: float  # Start time in seconds
    end: float  # End time in seconds
    confidence: float = 0.0
    text: str | None = None  # Optional transcription text

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.end - self.start


@dataclass
class DiarizationResult:
    """Result from speaker diarization.

    Attributes:
        segments: List of speaker segments with timing
        num_speakers: Number of speakers detected
        speaker_embeddings: Optional speaker embeddings for each speaker
        model_name: Name of the model used
        metadata: Additional metadata
    """

    segments: list[SpeakerSegment]
    num_speakers: int
    speaker_embeddings: dict[str, mx.array] | None = None
    model_name: str = ""
    metadata: dict = field(default_factory=dict)

    def to_rttm(self, filename: str = "audio") -> str:
        """Export as RTTM format (standard diarization format).

        Parameters
        ----------
        filename : str
            Filename to use in RTTM output.

        Returns
        -------
        str
            RTTM formatted string.
        """
        lines = []
        for seg in self.segments:
            # RTTM format: SPEAKER file 1 start duration <NA> <NA> spk <NA> <NA>
            duration = seg.end - seg.start
            line = (
                f"SPEAKER {filename} 1 {seg.start:.3f} {duration:.3f} "
                f"<NA> <NA> {seg.speaker} <NA> <NA>"
            )
            lines.append(line)
        return "\n".join(lines)

    def get_speaker_segments(self, speaker: str) -> list[SpeakerSegment]:
        """Get all segments for a specific speaker.

        Parameters
        ----------
        speaker : str
            Speaker ID (e.g., "SPEAKER_00").

        Returns
        -------
        list
            Segments for the specified speaker.
        """
        return [seg for seg in self.segments if seg.speaker == speaker]

    def get_speaker_duration(self, speaker: str) -> float:
        """Get total speaking duration for a speaker.

        Parameters
        ----------
        speaker : str
            Speaker ID.

        Returns
        -------
        float
            Total duration in seconds.
        """
        return sum(seg.duration for seg in self.get_speaker_segments(speaker))

    @property
    def speakers(self) -> list[str]:
        """List of unique speaker IDs."""
        return sorted({seg.speaker for seg in self.segments})

    @property
    def total_duration(self) -> float:
        """Total duration of all segments."""
        if not self.segments:
            return 0.0
        return max(seg.end for seg in self.segments)

    def save(self, path: str | Path, format: str = "rttm") -> Path:
        """Save diarization result to file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        format : str, default="rttm"
            Output format: "rttm" or "json".

        Returns
        -------
        Path
            Path to saved file.
        """
        import json

        path = Path(path)

        if format == "rttm":
            path.write_text(self.to_rttm(path.stem))
        elif format == "json":
            data = {
                "num_speakers": self.num_speakers,
                "segments": [
                    {
                        "speaker": s.speaker,
                        "start": s.start,
                        "end": s.end,
                        "text": s.text,
                    }
                    for s in self.segments
                ],
            }
            path.write_text(json.dumps(data, indent=2))
        else:
            raise ValueError(f"Unknown format: {format}")

        return path


@dataclass
class SpeechResult(AudioData):
    """Result from text-to-speech synthesis.

    Extends AudioData with TTS-specific metadata.

    Attributes:
        array: Generated audio data
        sample_rate: Sample rate
        text: Original text input
        description: Voice description used (if any)
        model_name: Name of the model used
        generation_params: Parameters used for generation
    """

    text: str = ""
    description: str | None = None
    model_name: str = ""
    generation_params: dict = field(default_factory=dict)

    def play(self) -> None:
        """Play the generated speech (requires sounddevice)."""
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "sounddevice is required for playback. "
                "Install with: pip install sounddevice"
            )
        sd.play(self.to_numpy().T, self.sample_rate)
        sd.wait()

    @property
    def duration(self) -> float:
        """Duration of the generated speech in seconds."""
        return self.array.shape[-1] / self.sample_rate


@dataclass
class EnhancementResult(AudioData):
    """Result from audio enhancement.

    Attributes:
        array: Enhanced audio data
        sample_rate: Sample rate
        model_name: Name of the model used
        snr_improvement: Estimated SNR improvement in dB (if available)
        metadata: Additional enhancement metadata
    """

    model_name: str = ""
    snr_improvement: float | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def before_after(self) -> tuple[mx.array, mx.array]:
        """Get original and enhanced arrays for comparison.

        Returns
        -------
        tuple
            (original, enhanced) arrays.

        Raises
        ------
        ValueError
            If original audio was not stored (keep_original=False).
        """
        original = self.metadata.get("original")
        if original is not None:
            return (original, self.array)
        raise ValueError(
            "Original audio not stored. "
            "Use keep_original=True when calling enhance()."
        )


@dataclass
class ClassificationResult:
    """Result from audio classification (single-label).

    Attributes:
        predicted_class: Predicted class index or name
        probabilities: Class probabilities [num_classes]
        class_names: Optional list of class names
        top_k_classes: Top-k class indices or names
        top_k_probs: Top-k probabilities
        model_name: Name of the model used
        metadata: Additional metadata
    """

    predicted_class: int | str
    probabilities: mx.array
    class_names: list[str] | None = None
    top_k_classes: list[int | str] | None = None
    top_k_probs: list[float] | None = None
    model_name: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        """Confidence of the top prediction."""
        import mlx.core as mx

        return float(mx.max(self.probabilities))

    def get_probability(self, class_name: str | int) -> float:
        """Get probability for a specific class.

        Args:
            class_name: Class name or index

        Returns:
            Probability for the class
        """
        if isinstance(class_name, str):
            if self.class_names is None:
                raise ValueError("No class names available")
            idx = self.class_names.index(class_name)
        else:
            idx = class_name
        return float(self.probabilities[idx])

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "top_k_classes": self.top_k_classes,
            "top_k_probs": self.top_k_probs,
            "class_names": self.class_names,
        }

    def __repr__(self) -> str:
        return (
            f"ClassificationResult(class={self.predicted_class!r}, "
            f"confidence={self.confidence:.2%})"
        )


@dataclass
class CLAPEmbeddingResult:
    """Result from CLAP embedding.

    Extends EmbeddingResult with audio-text similarity support.

    Attributes:
        audio_embeds: Audio embeddings [B, dim] or None
        text_embeds: Text embeddings [B, dim] or None
        similarity: Similarity matrix [B_audio, B_text] if both provided
        text_labels: Original text labels (for zero-shot)
        model_name: Name of the model used
        metadata: Additional metadata
    """

    audio_embeds: mx.array | None = None
    text_embeds: mx.array | None = None
    similarity: mx.array | None = None
    text_labels: list[str] | None = None
    model_name: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def vectors(self) -> mx.array:
        """Get primary embedding vectors (audio if available, else text)."""
        if self.audio_embeds is not None:
            return self.audio_embeds
        if self.text_embeds is not None:
            return self.text_embeds
        raise ValueError("No embeddings available")

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self.vectors.shape[-1]

    def best_match(self, top_k: int = 1) -> list[str] | str:
        """Get best matching text label(s) for audio.

        Requires both audio and text embeddings with text_labels.

        Args:
            top_k: Number of top matches to return

        Returns:
            Best matching label(s)
        """
        if self.similarity is None or self.text_labels is None:
            raise ValueError(
                "Need similarity matrix and text labels for best_match"
            )

        import mlx.core as mx

        # Get top-k indices
        if self.similarity.ndim == 1:
            sim = self.similarity
        else:
            sim = self.similarity[0]  # First audio sample

        indices = mx.argsort(sim)[::-1][:top_k]
        indices = [int(i) for i in indices]

        matches = [self.text_labels[i] for i in indices]
        return matches[0] if top_k == 1 else matches

    def to_numpy(self) -> np.ndarray:
        """Convert primary embeddings to NumPy array."""
        import numpy as np
        return np.array(self.vectors)

    def cosine_similarity(self, other: CLAPEmbeddingResult) -> float:
        """Compute cosine similarity with another embedding."""
        import mlx.core as mx

        a = self.vectors
        b = other.vectors

        # Handle batched case
        if a.ndim > 1:
            a = a[0]
        if b.ndim > 1:
            b = b[0]

        a = a / mx.linalg.norm(a)
        b = b / mx.linalg.norm(b)
        return float(mx.sum(a * b))


@dataclass
class TaggingResult:
    """Result from audio tagging (multi-label classification).

    Attributes:
        tags: List of active tag indices or names
        probabilities: Tag probabilities [num_tags]
        tag_names: Optional list of all tag names
        threshold: Threshold used for tagging
        model_name: Name of the model used
        metadata: Additional metadata
    """

    tags: list[int | str]
    probabilities: mx.array
    tag_names: list[str] | None = None
    threshold: float = 0.5
    model_name: str = ""
    metadata: dict = field(default_factory=dict)

    def get_probability(self, tag: str | int) -> float:
        """Get probability for a specific tag.

        Args:
            tag: Tag name or index

        Returns:
            Probability for the tag
        """
        if isinstance(tag, str):
            if self.tag_names is None:
                raise ValueError("No tag names available")
            idx = self.tag_names.index(tag)
        else:
            idx = tag
        return float(self.probabilities[idx])

    def top_k(self, k: int = 5) -> list[tuple[str | int, float]]:
        """Get top-k tags by probability.

        Args:
            k: Number of top tags to return

        Returns:
            List of (tag, probability) tuples
        """
        import mlx.core as mx

        sorted_indices = mx.argsort(self.probabilities)[::-1][:k]
        result = []
        for idx in sorted_indices:
            idx = int(idx)
            prob = float(self.probabilities[idx])
            name = self.tag_names[idx] if self.tag_names else idx
            result.append((name, prob))
        return result

    def above_threshold(self, threshold: float | None = None) -> list[tuple[str | int, float]]:
        """Get all tags above a threshold.

        Args:
            threshold: Probability threshold (default: use instance threshold)

        Returns:
            List of (tag, probability) tuples
        """
        if threshold is None:
            threshold = self.threshold

        result = []
        for idx, prob in enumerate(self.probabilities):
            prob_val = float(prob)
            if prob_val >= threshold:
                name = self.tag_names[idx] if self.tag_names else idx
                result.append((name, prob_val))

        # Sort by probability descending
        return sorted(result, key=lambda x: x[1], reverse=True)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "tags": self.tags,
            "threshold": self.threshold,
            "top_5": self.top_k(5),
            "tag_names": self.tag_names,
        }

    def __repr__(self) -> str:
        tags_str = ", ".join(str(t) for t in self.tags[:5])
        if len(self.tags) > 5:
            tags_str += f", ... ({len(self.tags)} total)"
        return f"TaggingResult(tags=[{tags_str}])"


__all__ = [
    # Base
    "Result",
    # Audio Result Types
    "AudioData",
    "SeparationResult",
    "GenerationResult",
    "SpeechResult",
    "EnhancementResult",
    # Text Result Types
    "TranscriptionResult",
    "TranscriptionSegment",
    # Speaker Result Types
    "DiarizationResult",
    "SpeakerSegment",
    # Embedding Result Types
    "EmbeddingResult",
    "CLAPEmbeddingResult",
    # Classification Result Types
    "ClassificationResult",
    "TaggingResult",
]
