"""E-GMD (Expanded Groove MIDI Dataset) loader for MLX.

E-GMD contains 444 hours of human drum performances with MIDI annotations.
Audio is synthesized from MIDI using 43 different drum kits.

Ported from PyTorch implementation.
"""

import csv
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterator

import mlx.core as mx
import numpy as np

from mlx_audio.models.drums.config import SpectrogramConfig


class DrumClass(IntEnum):
    """14-class drum taxonomy with cymbal granularity."""

    KICK = 0
    SNARE_CENTER = 1
    SNARE_RIMSHOT = 2
    SNARE_CROSSSTICK = 3
    HIHAT_CLOSED = 4
    HIHAT_OPEN = 5
    HIHAT_PEDAL = 6
    RIDE_BOW = 7
    RIDE_BELL = 8
    CRASH = 9
    TOM_HIGH = 10
    TOM_MID = 11
    TOM_LOW = 12
    EFFECTS = 13


NUM_CLASSES = 14

# Human-readable names
DRUM_CLASS_NAMES = [
    "Kick",
    "Snare (center)",
    "Snare (rimshot)",
    "Snare (cross-stick)",
    "Hi-hat (closed)",
    "Hi-hat (open)",
    "Hi-hat (pedal)",
    "Ride (bow)",
    "Ride (bell)",
    "Crash",
    "Tom (high)",
    "Tom (mid)",
    "Tom (low/floor)",
    "Effects (china/splash)",
]

# Extended MIDI mappings for E-GMD (Roland TD-17 note numbers)
EXTENDED_MIDI_MAPPING: dict[int, DrumClass] = {
    # Kicks
    35: DrumClass.KICK,
    36: DrumClass.KICK,
    # Snares
    38: DrumClass.SNARE_CENTER,
    40: DrumClass.SNARE_RIMSHOT,
    37: DrumClass.SNARE_CROSSSTICK,
    # Hi-hats
    22: DrumClass.HIHAT_CLOSED,
    42: DrumClass.HIHAT_CLOSED,
    26: DrumClass.HIHAT_OPEN,
    46: DrumClass.HIHAT_OPEN,
    44: DrumClass.HIHAT_PEDAL,
    # Rides
    51: DrumClass.RIDE_BOW,
    59: DrumClass.RIDE_BOW,
    53: DrumClass.RIDE_BELL,
    # Crashes
    49: DrumClass.CRASH,
    55: DrumClass.CRASH,
    57: DrumClass.CRASH,
    52: DrumClass.CRASH,
    # Toms
    48: DrumClass.TOM_HIGH,
    50: DrumClass.TOM_HIGH,
    45: DrumClass.TOM_MID,
    47: DrumClass.TOM_MID,
    43: DrumClass.TOM_LOW,
    41: DrumClass.TOM_LOW,
    58: DrumClass.TOM_LOW,
    # Effects
    54: DrumClass.EFFECTS,
    56: DrumClass.EFFECTS,
}


@dataclass
class DrumHit:
    """A single drum hit event."""

    time: float  # Time in seconds
    drum_class: DrumClass
    velocity: int  # 0-127


@dataclass
class EGMDExample:
    """A single example from E-GMD dataset."""

    midi_path: Path
    audio_path: Path
    drummer: str
    session: str
    style: str
    tempo: float
    time_signature: str
    duration: float
    kit_name: str

    def load_hits(self) -> list[DrumHit]:
        """Load drum hits from MIDI file."""
        return load_midi_hits(self.midi_path)

    def load_spectrogram(self) -> np.ndarray:
        """Load spectrogram from cache.

        Returns:
            Numpy array of shape (n_mels, time_frames)
        """
        import torch

        cache_path = self.audio_path.with_suffix(".spec.pt")

        if cache_path.exists():
            spec = torch.load(cache_path, map_location="cpu", weights_only=True)
            return spec.numpy()

        raise FileNotFoundError(
            f"Cached spectrogram not found at {cache_path}. "
            "Run precompute_spectrograms.py first."
        )


def load_midi_hits(midi_path: Path | str) -> list[DrumHit]:
    """Load drum hits from a MIDI file.

    Args:
        midi_path: Path to MIDI file

    Returns:
        List of DrumHit objects sorted by time
    """
    import mido

    midi = mido.MidiFile(midi_path)
    hits = []

    # Track absolute time
    current_time = 0.0

    for msg in midi:
        current_time += msg.time

        if msg.type == "note_on" and msg.velocity > 0:
            # Map MIDI note to our drum class
            if msg.note in EXTENDED_MIDI_MAPPING:
                drum_class = EXTENDED_MIDI_MAPPING[msg.note]
                hit = DrumHit(
                    time=current_time,
                    drum_class=drum_class,
                    velocity=msg.velocity,
                )
                hits.append(hit)

    # Sort by time
    hits.sort(key=lambda h: h.time)

    return hits


def create_onset_target(
    hits: list[DrumHit],
    num_frames: int,
    config: SpectrogramConfig,
    num_classes: int = 14,
) -> tuple[np.ndarray, np.ndarray]:
    """Create onset and velocity target arrays from drum hits.

    Args:
        hits: List of DrumHit objects with time, drum_class, velocity
        num_frames: Number of spectrogram frames
        config: Spectrogram configuration for time-to-frame conversion
        num_classes: Number of drum classes

    Returns:
        Tuple of:
            - onset_target: Binary array (num_frames, num_classes)
            - velocity_target: Velocity array (num_frames, num_classes), 0-1 normalized
    """
    onset_target = np.zeros((num_frames, num_classes), dtype=np.float32)
    velocity_target = np.zeros((num_frames, num_classes), dtype=np.float32)

    for hit in hits:
        frame = config.time_to_frame(hit.time)
        if 0 <= frame < num_frames:
            class_idx = int(hit.drum_class)
            onset_target[frame, class_idx] = 1.0
            velocity_target[frame, class_idx] = hit.velocity / 127.0

    return onset_target, velocity_target


def parse_egmd_csv(csv_path: Path) -> list[dict]:
    """Parse E-GMD metadata CSV."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def iter_egmd_examples(
    dataset_dir: Path,
    split: str | None = None,
) -> Iterator[EGMDExample]:
    """Iterate over E-GMD examples.

    Args:
        dataset_dir: Path to e-gmd-v1.0.0 directory
        split: Optional split filter ('train', 'validation', 'test')

    Yields:
        EGMDExample objects
    """
    csv_path = dataset_dir / "e-gmd-v1.0.0.csv"
    if not csv_path.exists():
        # Try finding CSV in subdirectory
        subdirs = list(dataset_dir.glob("e-gmd-*"))
        if subdirs:
            csv_path = subdirs[0] / "e-gmd-v1.0.0.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"E-GMD CSV not found at {csv_path}")

    base_dir = csv_path.parent
    metadata = parse_egmd_csv(csv_path)

    for row in metadata:
        # Filter by split if specified
        if split and row.get("split") != split:
            continue

        midi_path = base_dir / row["midi_filename"]
        audio_path = base_dir / row["audio_filename"]
        spec_path = audio_path.with_suffix(".spec.pt")

        # Skip if MIDI doesn't exist
        if not midi_path.exists():
            continue

        # Skip if precomputed spectrogram doesn't exist
        if not spec_path.exists():
            continue

        yield EGMDExample(
            midi_path=midi_path,
            audio_path=audio_path,
            drummer=row.get("drummer", "unknown"),
            session=row.get("session", "unknown"),
            style=row.get("style", "unknown"),
            tempo=float(row.get("bpm", 120)),
            time_signature=row.get("time_signature", "4-4"),
            duration=float(row.get("duration", 0)),
            kit_name=row.get("kit_name", "unknown"),
        )


class EGMDDataset:
    """Dataset for E-GMD.

    Loads audio as mel-spectrograms with onset/velocity targets.
    Compatible with mlx-audio DataLoader.
    """

    def __init__(
        self,
        dataset_dir: Path | str,
        split: str | None = None,
        config: SpectrogramConfig | None = None,
        seq_length: int = 512,  # Number of frames per sample
        stride: int = 256,  # Stride between samples (for overlapping)
    ):
        """Initialize E-GMD dataset.

        Args:
            dataset_dir: Path to E-GMD dataset
            split: Data split ('train', 'validation', 'test')
            config: Spectrogram configuration
            seq_length: Number of frames per training sample
            stride: Stride between consecutive samples
        """
        self.dataset_dir = Path(dataset_dir)
        self.config = config or SpectrogramConfig()
        self.seq_length = seq_length
        self.stride = stride

        # Load all examples
        self.examples = list(iter_egmd_examples(self.dataset_dir, split))

        if len(self.examples) == 0:
            raise ValueError(
                f"No examples found in {dataset_dir} for split={split}. "
                "Make sure the dataset exists and spectrograms are precomputed."
            )

        # Pre-compute sample indices
        # Each example can produce multiple training samples
        self.sample_indices: list[tuple[int, int]] = []  # (example_idx, frame_start)
        self._compute_sample_indices()

    def _compute_sample_indices(self) -> None:
        """Pre-compute indices for all training samples."""
        for ex_idx, example in enumerate(self.examples):
            # Estimate number of frames from duration
            num_frames = self.config.time_to_frame(example.duration)

            # Generate sample start positions
            for start in range(0, max(1, num_frames - self.seq_length), self.stride):
                self.sample_indices.append((ex_idx, start))

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Get a training sample.

        Returns:
            Dict with keys:
                - spectrogram: (1, seq_length, n_mels) mel-spectrogram
                - onset_target: (seq_length, num_classes) binary onsets
                - velocity_target: (seq_length, num_classes) velocities 0-1
        """
        ex_idx, frame_start = self.sample_indices[idx]
        example = self.examples[ex_idx]

        # Load full spectrogram
        full_spec = example.load_spectrogram()  # (n_mels, time)

        # Extract window
        frame_end = frame_start + self.seq_length
        if frame_end > full_spec.shape[1]:
            # Pad if necessary
            spec = np.zeros((self.config.n_mels, self.seq_length), dtype=np.float32)
            available = full_spec.shape[1] - frame_start
            spec[:, :available] = full_spec[:, frame_start:]
        else:
            spec = full_spec[:, frame_start:frame_end]

        # Transpose to (seq_length, n_mels) for model input
        spec = spec.T.astype(np.float32)

        # Load hits and create targets
        hits = example.load_hits()

        # Filter hits to this time window
        start_time = self.config.frame_to_time(frame_start)
        end_time = self.config.frame_to_time(frame_end)
        window_hits = [
            DrumHit(
                time=h.time - start_time,  # Relative time within window
                drum_class=h.drum_class,
                velocity=h.velocity,
            )
            for h in hits
            if start_time <= h.time < end_time
        ]

        onset_target, velocity_target = create_onset_target(
            window_hits,
            self.seq_length,
            self.config,
            NUM_CLASSES,
        )

        return {
            "spectrogram": spec[np.newaxis, :, :],  # (1, seq_length, n_mels)
            "onset_target": onset_target,
            "velocity_target": velocity_target,
        }


def collate_fn(batch: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Collate function for DataLoader.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched dict with stacked arrays
    """
    return {
        "spectrogram": np.stack([b["spectrogram"] for b in batch]),
        "onset_target": np.stack([b["onset_target"] for b in batch]),
        "velocity_target": np.stack([b["velocity_target"] for b in batch]),
    }


def to_mlx(batch: dict[str, np.ndarray]) -> dict[str, mx.array]:
    """Convert numpy batch to MLX arrays.

    Args:
        batch: Dict with numpy arrays

    Returns:
        Dict with MLX arrays
    """
    return {k: mx.array(v) for k, v in batch.items()}


def get_egmd_dataloader(
    dataset_dir: Path | str,
    split: str,
    batch_size: int = 32,
    shuffle: bool = True,
    **kwargs,
) -> Any:
    """Create a DataLoader for E-GMD.

    Args:
        dataset_dir: Path to E-GMD dataset
        split: Data split
        batch_size: Batch size
        shuffle: Whether to shuffle
        **kwargs: Additional arguments for EGMDDataset

    Returns:
        DataLoader instance
    """
    from mlx_audio.data import DataLoader

    dataset = EGMDDataset(dataset_dir, split=split, **kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        mlx_transforms=to_mlx,
    )
