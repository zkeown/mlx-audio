"""File-based audio sink for streaming."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    pass


class FileSink:
    """Audio sink that writes to a file.

    Accumulates audio samples and writes to a file on stop().
    For source separation models, can select a specific stem.

    Args:
        path: Path to the output audio file
        sample_rate: Sample rate for the output file
        stem_index: For multi-stem output, which stem to write (default: None = all/first)
        stem_name: Alternative to stem_index, specify stem by name
        format: Output format (default: inferred from extension)
        subtype: Audio subtype (e.g., 'PCM_16', 'FLOAT')

    Example:
        >>> sink = FileSink("output.wav", sample_rate=44100)
        >>> sink.start()
        >>> for chunk in audio_chunks:
        ...     sink.write(chunk)
        >>> sink.stop()  # File is written here
    """

    def __init__(
        self,
        path: str | Path,
        sample_rate: int = 44100,
        stem_index: int | None = None,
        stem_name: str | None = None,
        format: str | None = None,
        subtype: str | None = None,
    ) -> None:
        self._path = Path(path)
        self._sample_rate = sample_rate
        self._stem_index = stem_index
        self._stem_name = stem_name
        self._format = format
        self._subtype = subtype

        # State
        self._chunks: list[np.ndarray] = []
        self._started: bool = False

    def start(self) -> None:
        """Prepare for writing."""
        if self._started:
            return
        self._chunks = []
        self._started = True

    def stop(self) -> None:
        """Finalize and write the audio file."""
        if not self._started:
            return

        if self._chunks:
            self._write_file()

        self._chunks = []
        self._started = False

    def write(self, audio: mx.array | np.ndarray) -> bool:
        """Write audio samples to the sink.

        Args:
            audio: Audio samples. Shape can be:
                - [channels, samples] for regular audio
                - [sources, channels, samples] for separated stems

        Returns:
            True if write succeeded
        """
        if not self._started:
            raise RuntimeError("Sink not started. Call start() first.")

        # Convert to numpy
        if isinstance(audio, mx.array):
            audio = np.array(audio)

        # Handle separated stems [sources, channels, samples]
        if audio.ndim == 3:
            if self._stem_index is not None:
                audio = audio[self._stem_index]
            else:
                # Default to first stem if not specified
                audio = audio[0]

        self._chunks.append(audio)
        return True

    def _write_file(self) -> None:
        """Write accumulated audio to file."""
        if not self._chunks:
            return

        # Concatenate all chunks
        audio = np.concatenate(self._chunks, axis=-1)

        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import soundfile as sf

            # soundfile expects [samples, channels]
            if audio.ndim == 2:
                audio = audio.T
            elif audio.ndim == 1:
                audio = audio.reshape(-1, 1)

            sf.write(
                str(self._path),
                audio,
                self._sample_rate,
                format=self._format,
                subtype=self._subtype,
            )
        except ImportError:
            raise ImportError(
                "soundfile is required for FileSink. "
                "Install with: pip install soundfile"
            )

    @property
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""
        return self._sample_rate

    @property
    def samples_written(self) -> int:
        """Total samples written so far."""
        if not self._chunks:
            return 0
        return sum(chunk.shape[-1] for chunk in self._chunks)

    @property
    def duration_written(self) -> float:
        """Duration of audio written in seconds."""
        return self.samples_written / self._sample_rate

    def __enter__(self) -> FileSink:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()


class MultiFileSink:
    """Audio sink that writes multiple stems to separate files.

    Useful for source separation models that output multiple stems.

    Args:
        output_dir: Directory to write output files
        sample_rate: Sample rate for output files
        stem_names: Names for each stem (used in filenames)
        prefix: Prefix for output filenames
        format: Output format (default: wav)

    Example:
        >>> sink = MultiFileSink(
        ...     "output/",
        ...     stem_names=["drums", "bass", "other", "vocals"],
        ...     sample_rate=44100,
        ... )
        >>> sink.start()
        >>> sink.write(separated_audio)  # [4, 2, samples]
        >>> sink.stop()
        # Creates: output/drums.wav, output/bass.wav, etc.
    """

    def __init__(
        self,
        output_dir: str | Path,
        sample_rate: int = 44100,
        stem_names: list[str] | None = None,
        prefix: str = "",
        format: str = "wav",
    ) -> None:
        self._output_dir = Path(output_dir)
        self._sample_rate = sample_rate
        self._stem_names = stem_names or ["stem_0", "stem_1", "stem_2", "stem_3"]
        self._prefix = prefix
        self._format = format

        # State
        self._sinks: list[FileSink] = []
        self._started: bool = False

    def start(self) -> None:
        """Prepare for writing."""
        if self._started:
            return

        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._sinks = []
        for name in self._stem_names:
            filename = f"{self._prefix}{name}.{self._format}"
            sink = FileSink(
                self._output_dir / filename,
                sample_rate=self._sample_rate,
            )
            sink.start()
            self._sinks.append(sink)

        self._started = True

    def stop(self) -> None:
        """Finalize and write all files."""
        for sink in self._sinks:
            sink.stop()
        self._sinks = []
        self._started = False

    def write(self, audio: mx.array | np.ndarray) -> bool:
        """Write separated audio to multiple files.

        Args:
            audio: Separated audio with shape [sources, channels, samples]

        Returns:
            True if write succeeded
        """
        if not self._started:
            raise RuntimeError("Sink not started. Call start() first.")

        # Convert to numpy
        if isinstance(audio, mx.array):
            audio = np.array(audio)

        if audio.ndim != 3:
            raise ValueError(
                f"Expected 3D array [sources, channels, samples], got shape {audio.shape}"
            )

        num_sources = audio.shape[0]
        if num_sources > len(self._sinks):
            raise ValueError(
                f"Audio has {num_sources} sources but only {len(self._sinks)} sinks"
            )

        for i, sink in enumerate(self._sinks[:num_sources]):
            sink.write(audio[i])

        return True

    @property
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""
        return self._sample_rate

    def __enter__(self) -> MultiFileSink:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
