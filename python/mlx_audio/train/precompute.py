"""Data pre-computation utilities for mlx-train.

Pre-compute expensive transformations (spectrograms, embeddings, etc.) ahead
of training to speed up data loading.

Example usage:
    >>> from mlx_audio.train.precompute import precompute_dataset, Preprocessor
    >>>
    >>> class SpectrogramPreprocessor(Preprocessor):
    ...     def __init__(self, n_fft=2048, hop_length=512, n_mels=128):
    ...         self.n_fft = n_fft
    ...         self.hop_length = hop_length
    ...         self.n_mels = n_mels
    ...
    ...     def process(self, input_path: Path) -> np.ndarray:
    ...         import librosa
    ...         audio, sr = librosa.load(input_path, sr=44100)
    ...         spec = librosa.feature.melspectrogram(
    ...             y=audio, sr=sr, n_fft=self.n_fft,
    ...             hop_length=self.hop_length, n_mels=self.n_mels
    ...         )
    ...         return librosa.power_to_db(spec, ref=np.max)
    ...
    ...     def get_output_path(self, input_path: Path) -> Path:
    ...         return input_path.with_suffix(".spec.npy")
    >>>
    >>> # Pre-compute all spectrograms
    >>> audio_files = list(Path("data").glob("**/*.wav"))
    >>> stats = precompute_dataset(audio_files, SpectrogramPreprocessor())
    >>> print(f"Computed {stats.computed}, skipped {stats.skipped}")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass
class PrecomputeStats:
    """Statistics from a precompute run."""

    total: int
    computed: int
    skipped: int
    errors: int
    error_messages: list[tuple[str, str]]


class Preprocessor(ABC):
    """Base class for data preprocessors.

    Subclass this and implement process() and get_output_path() to define
    your preprocessing logic.
    """

    @abstractmethod
    def process(self, input_path: Path) -> np.ndarray:
        """Process a single input file.

        Args:
            input_path: Path to the input file

        Returns:
            Processed data as numpy array
        """
        pass

    @abstractmethod
    def get_output_path(self, input_path: Path) -> Path:
        """Get the output path for a processed file.

        Args:
            input_path: Path to the input file

        Returns:
            Path where the processed data should be saved
        """
        pass

    def should_skip(self, input_path: Path, output_path: Path) -> bool:
        """Check if processing should be skipped (e.g., output already exists).

        Args:
            input_path: Path to the input file
            output_path: Path where output would be saved

        Returns:
            True if processing should be skipped
        """
        return output_path.exists()

    def save(self, data: np.ndarray, output_path: Path) -> None:
        """Save processed data to disk.

        Args:
            data: Processed numpy array
            output_path: Path to save to

        Override for custom save formats.
        """
        np.save(output_path, data)


def _process_single_file(
    args: tuple[Path, Preprocessor, bool]
) -> tuple[str, bool, str]:
    """Process a single file (runs in worker process).

    Args:
        args: Tuple of (input_path, preprocessor, force)

    Returns:
        Tuple of (input_path, success, message)
    """
    input_path, preprocessor, force = args

    try:
        output_path = preprocessor.get_output_path(input_path)

        # Skip if already processed
        if not force and preprocessor.should_skip(input_path, output_path):
            return str(input_path), True, "skipped"

        # Check input exists
        if not input_path.exists():
            return str(input_path), False, "input not found"

        # Process
        data = preprocessor.process(input_path)

        # Save
        preprocessor.save(data, output_path)

        return str(input_path), True, f"saved ({data.shape})"

    except Exception as e:
        return str(input_path), False, str(e)


def precompute_dataset(
    input_files: list[Path],
    preprocessor: Preprocessor,
    *,
    num_workers: int = 4,
    force: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
    verbose: bool = True,
) -> PrecomputeStats:
    """Pre-compute transformations for a dataset in parallel.

    Args:
        input_files: List of input file paths to process
        preprocessor: Preprocessor instance defining the transformation
        num_workers: Number of parallel workers
        force: If True, recompute even if output exists
        progress_callback: Optional callback(current, total) for progress updates
        verbose: If True, print progress and errors

    Returns:
        PrecomputeStats with counts and any error messages
    """
    total = len(input_files)
    computed = 0
    skipped = 0
    errors = 0
    error_messages: list[tuple[str, str]] = []

    if total == 0:
        return PrecomputeStats(
            total=0, computed=0, skipped=0, errors=0, error_messages=[]
        )

    # Prepare work items
    work_items = [(path, preprocessor, force) for path in input_files]

    # Process in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_process_single_file, item): item
            for item in work_items
        }

        for i, future in enumerate(as_completed(futures)):
            input_path, success, message = future.result()

            if success:
                if message == "skipped":
                    skipped += 1
                else:
                    computed += 1
            else:
                errors += 1
                error_messages.append((input_path, message))
                if verbose:
                    print(f"  Error: {input_path}: {message}")

            # Progress callback
            if progress_callback:
                progress_callback(i + 1, total)
            elif verbose and (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{total}")

    return PrecomputeStats(
        total=total,
        computed=computed,
        skipped=skipped,
        errors=errors,
        error_messages=error_messages,
    )


# ==================== Common Preprocessors ====================


class MelSpectrogramPreprocessor(Preprocessor):
    """Preprocessor that computes mel spectrograms from audio files.

    Uses librosa for computation. Output is saved as .spec.npy files.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        fmin: float = 20.0,
        fmax: float = 20000.0,
        power: float = 2.0,
        normalize: bool = True,
        suffix: str = ".spec.npy",
    ):
        """Initialize the mel spectrogram preprocessor.

        Args:
            sample_rate: Target sample rate for loading audio
            n_fft: FFT window size
            hop_length: Hop length between frames
            n_mels: Number of mel bands
            fmin: Minimum frequency for mel filterbank
            fmax: Maximum frequency for mel filterbank
            power: Exponent for mel spectrogram (2.0 = power, 1.0 = magnitude)
            normalize: If True, normalize to [-1, 1] range
            suffix: Output file suffix
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.power = power
        self.normalize = normalize
        self.suffix = suffix

    def process(self, input_path: Path) -> np.ndarray:
        """Compute mel spectrogram from audio file."""
        import librosa

        # Load audio
        audio, _ = librosa.load(str(input_path), sr=self.sample_rate, mono=True)

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=self.power,
        )

        # Convert to log scale (dB)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to [-1, 1] range
        if self.normalize:
            log_mel = (log_mel + 80) / 80  # Assuming -80 dB floor
            log_mel = np.clip(log_mel, -1, 1)

        return log_mel.astype(np.float32)

    def get_output_path(self, input_path: Path) -> Path:
        """Get output path by replacing suffix."""
        return input_path.with_suffix(self.suffix)


class EmbeddingPreprocessor(Preprocessor):
    """Preprocessor that computes embeddings using a model.

    Useful for pre-computing CLAP embeddings, speaker embeddings, etc.
    """

    def __init__(
        self,
        embed_fn: Callable[[Path], np.ndarray],
        suffix: str = ".emb.npy",
    ):
        """Initialize the embedding preprocessor.

        Args:
            embed_fn: Function that takes a path and returns embeddings
            suffix: Output file suffix
        """
        self.embed_fn = embed_fn
        self.suffix = suffix

    def process(self, input_path: Path) -> np.ndarray:
        """Compute embedding from input file."""
        return self.embed_fn(input_path)

    def get_output_path(self, input_path: Path) -> Path:
        """Get output path by replacing suffix."""
        return input_path.with_suffix(self.suffix)


# ==================== CLI Utility ====================


def run_precompute_cli(
    input_files: list[Path],
    preprocessor: Preprocessor,
    description: str = "Pre-computing dataset",
    num_workers: int = 4,
    force: bool = False,
) -> PrecomputeStats:
    """Run pre-computation with CLI output.

    Args:
        input_files: List of input file paths
        preprocessor: Preprocessor instance
        description: Description for progress output
        num_workers: Number of parallel workers
        force: If True, recompute even if output exists

    Returns:
        PrecomputeStats with results
    """
    print("=" * 60)
    print(description)
    print("=" * 60)

    print(f"\nFound {len(input_files)} files")

    # Count existing
    existing = sum(
        1 for f in input_files
        if preprocessor.get_output_path(f).exists()
    )
    print(f"Already computed: {existing}")

    if existing == len(input_files) and not force:
        print("\nAll files already computed! Use --force to recompute.")
        return PrecomputeStats(
            total=len(input_files),
            computed=0,
            skipped=len(input_files),
            errors=0,
            error_messages=[],
        )

    print(f"\nProcessing with {num_workers} workers...")

    stats = precompute_dataset(
        input_files,
        preprocessor,
        num_workers=num_workers,
        force=force,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Completed!")
    print(f"  Computed: {stats.computed}")
    print(f"  Skipped: {stats.skipped}")
    print(f"  Errors: {stats.errors}")
    print("=" * 60)

    return stats
