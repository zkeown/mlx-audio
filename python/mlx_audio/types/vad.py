"""Voice Activity Detection result types."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx


@dataclass
class SpeechSegment:
    """A segment of detected speech with timing.

    Attributes:
        start: Start time in seconds
        end: End time in seconds
        probability: Average speech probability for this segment
    """

    start: float
    end: float
    probability: float = 0.0

    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end - self.start


@dataclass
class VADResult:
    """Result from voice activity detection.

    Provides detected speech segments and optional per-frame probabilities.

    Attributes:
        segments: List of detected speech segments
        probabilities: Per-frame speech probabilities (optional)
        sample_rate: Audio sample rate used for detection
        window_size_samples: Window size used for frame-level detection
        model_name: Name of the VAD model used
        threshold: Speech probability threshold used
        metadata: Additional metadata
    """

    segments: list[SpeechSegment]
    probabilities: "mx.array | None" = None
    sample_rate: int = 16000
    window_size_samples: int = 512
    model_name: str = ""
    threshold: float = 0.5
    metadata: dict = field(default_factory=dict)

    @property
    def speech_ratio(self) -> float:
        """Ratio of speech to total audio duration."""
        if not self.segments:
            return 0.0
        total_speech = sum(seg.duration for seg in self.segments)
        total_duration = self.total_duration
        if total_duration == 0:
            return 0.0
        return total_speech / total_duration

    @property
    def total_duration(self) -> float:
        """Total audio duration in seconds (estimated from segments or probabilities)."""
        if self.probabilities is not None:
            num_frames = self.probabilities.shape[0]
            return (num_frames * self.window_size_samples) / self.sample_rate
        if self.segments:
            return max(seg.end for seg in self.segments)
        return 0.0

    @property
    def num_segments(self) -> int:
        """Number of detected speech segments."""
        return len(self.segments)

    def get_speech_times(self) -> list[tuple[float, float]]:
        """Get speech segment times as list of (start, end) tuples.

        Returns:
            List of (start_seconds, end_seconds) tuples
        """
        return [(seg.start, seg.end) for seg in self.segments]

    def get_silence_times(self) -> list[tuple[float, float]]:
        """Get silence segment times as list of (start, end) tuples.

        Returns:
            List of (start_seconds, end_seconds) tuples
        """
        if not self.segments:
            return [(0.0, self.total_duration)] if self.total_duration > 0 else []

        silences = []
        prev_end = 0.0

        for seg in sorted(self.segments, key=lambda s: s.start):
            if seg.start > prev_end:
                silences.append((prev_end, seg.start))
            prev_end = seg.end

        # Add trailing silence
        if prev_end < self.total_duration:
            silences.append((prev_end, self.total_duration))

        return silences

    def get_speech_audio(self, audio: "mx.array") -> "mx.array":
        """Extract speech portions from audio.

        Args:
            audio: Audio array [samples] or [channels, samples]

        Returns:
            Concatenated speech portions
        """
        import mlx.core as mx

        if audio.ndim == 1:
            audio = audio[None, :]

        segments_audio = []
        for seg in self.segments:
            start_sample = int(seg.start * self.sample_rate)
            end_sample = int(seg.end * self.sample_rate)
            end_sample = min(end_sample, audio.shape[-1])
            if start_sample < end_sample:
                segments_audio.append(audio[:, start_sample:end_sample])

        if not segments_audio:
            return mx.array([])

        return mx.concatenate(segments_audio, axis=-1)

    def get_silence_audio(self, audio: "mx.array") -> "mx.array":
        """Extract silence portions from audio.

        Args:
            audio: Audio array [samples] or [channels, samples]

        Returns:
            Concatenated silence portions
        """
        import mlx.core as mx

        if audio.ndim == 1:
            audio = audio[None, :]

        silences = self.get_silence_times()
        segments_audio = []

        for start, end in silences:
            start_sample = int(start * self.sample_rate)
            end_sample = int(end * self.sample_rate)
            end_sample = min(end_sample, audio.shape[-1])
            if start_sample < end_sample:
                segments_audio.append(audio[:, start_sample:end_sample])

        if not segments_audio:
            return mx.array([])

        return mx.concatenate(segments_audio, axis=-1)

    def to_whisper_segments(self) -> list[dict]:
        """Convert to format compatible with Whisper preprocessing.

        Returns:
            List of dicts with 'start' and 'end' keys in seconds
        """
        return [
            {"start": seg.start, "end": seg.end}
            for seg in self.segments
        ]

    def to_audacity_labels(self) -> str:
        """Export as Audacity label track format.

        Returns:
            Tab-separated label track string
        """
        lines = []
        for i, seg in enumerate(self.segments):
            lines.append(f"{seg.start:.6f}\t{seg.end:.6f}\tspeech_{i+1}")
        return "\n".join(lines)

    def save(
        self,
        path: str | Path,
        format: str = "json",
    ) -> Path:
        """Save VAD result to file.

        Args:
            path: Output file path
            format: Output format ('json', 'txt', 'audacity')

        Returns:
            Path to saved file
        """
        path = Path(path)

        if format == "json":
            data = {
                "segments": [
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "probability": seg.probability,
                    }
                    for seg in self.segments
                ],
                "sample_rate": self.sample_rate,
                "threshold": self.threshold,
                "model_name": self.model_name,
                "speech_ratio": self.speech_ratio,
            }
            path.write_text(json.dumps(data, indent=2))

        elif format == "txt":
            lines = []
            for seg in self.segments:
                lines.append(f"{seg.start:.3f} - {seg.end:.3f} (prob: {seg.probability:.2f})")
            path.write_text("\n".join(lines))

        elif format == "audacity":
            path.write_text(self.to_audacity_labels())

        else:
            raise ValueError(f"Unknown format: {format}. Use 'json', 'txt', or 'audacity'")

        return path

    @classmethod
    def from_probabilities(
        cls,
        probabilities: "mx.array",
        sample_rate: int = 16000,
        window_size_samples: int = 512,
        threshold: float = 0.5,
        min_speech_duration: float = 0.25,
        min_silence_duration: float = 0.1,
        model_name: str = "",
    ) -> "VADResult":
        """Create VADResult from per-frame probabilities.

        Args:
            probabilities: Per-frame speech probabilities [num_frames]
            sample_rate: Audio sample rate
            window_size_samples: Samples per frame
            threshold: Speech probability threshold
            min_speech_duration: Minimum speech segment duration in seconds
            min_silence_duration: Minimum silence duration to split segments
            model_name: Model name

        Returns:
            VADResult with detected segments
        """
        probs_list = probabilities.tolist()
        window_duration = window_size_samples / sample_rate

        # Convert to segments
        segments = []
        in_speech = False
        speech_start = 0.0
        speech_probs = []

        for i, p in enumerate(probs_list):
            time = i * window_duration

            if p > threshold and not in_speech:
                in_speech = True
                speech_start = time
                speech_probs = [p]
            elif p > threshold and in_speech:
                speech_probs.append(p)
            elif p <= threshold and in_speech:
                in_speech = False
                avg_prob = sum(speech_probs) / len(speech_probs) if speech_probs else 0.0
                duration = time - speech_start + window_duration
                if duration >= min_speech_duration:
                    segments.append(SpeechSegment(
                        start=speech_start,
                        end=time + window_duration,
                        probability=avg_prob,
                    ))

        # Handle speech extending to end
        if in_speech:
            avg_prob = sum(speech_probs) / len(speech_probs) if speech_probs else 0.0
            end_time = len(probs_list) * window_duration
            duration = end_time - speech_start
            if duration >= min_speech_duration:
                segments.append(SpeechSegment(
                    start=speech_start,
                    end=end_time,
                    probability=avg_prob,
                ))

        # Merge segments separated by short silences
        if min_silence_duration > 0 and len(segments) > 1:
            merged = [segments[0]]
            for seg in segments[1:]:
                gap = seg.start - merged[-1].end
                if gap < min_silence_duration:
                    # Merge with previous segment
                    merged[-1] = SpeechSegment(
                        start=merged[-1].start,
                        end=seg.end,
                        probability=(merged[-1].probability + seg.probability) / 2,
                    )
                else:
                    merged.append(seg)
            segments = merged

        return cls(
            segments=segments,
            probabilities=probabilities,
            sample_rate=sample_rate,
            window_size_samples=window_size_samples,
            model_name=model_name,
            threshold=threshold,
        )

    def __repr__(self) -> str:
        return (
            f"VADResult(num_segments={self.num_segments}, "
            f"speech_ratio={self.speech_ratio:.1%}, "
            f"sample_rate={self.sample_rate})"
        )


__all__ = ["VADResult", "SpeechSegment"]
