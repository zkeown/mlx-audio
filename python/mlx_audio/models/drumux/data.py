"""E-GMD dataset loader for MLX training.

Loads E-GMD drum transcription dataset and yields batches compatible
with MLX training.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import numpy as np
import mido


# Drum class definitions (matching PyTorch version)
NUM_CLASSES = 14

# Extended MIDI mappings for E-GMD (Roland TD-17)
EXTENDED_MIDI_MAPPING: dict[int, int] = {
    # Kicks
    35: 0,  # Acoustic Bass Drum -> KICK
    36: 0,  # Bass Drum 1 -> KICK
    # Snares
    38: 1,  # Snare head -> SNARE_CENTER
    40: 2,  # Snare rim -> SNARE_RIMSHOT
    37: 3,  # Side Stick -> SNARE_CROSSSTICK
    # Hi-hats (TD-17 has bow and edge zones)
    22: 4,  # Hi-hat closed EDGE -> HIHAT_CLOSED
    42: 4,  # Hi-hat closed BOW -> HIHAT_CLOSED
    26: 5,  # Hi-hat open EDGE -> HIHAT_OPEN
    46: 5,  # Hi-hat open BOW -> HIHAT_OPEN
    44: 6,  # Pedal Hi-Hat -> HIHAT_PEDAL
    # Rides
    51: 7,  # Ride Cymbal 1 bow -> RIDE_BOW
    59: 7,  # Ride Cymbal 2 / edge -> RIDE_BOW
    53: 8,  # Ride Bell -> RIDE_BELL
    # Crashes (TD-17 bow and edge zones)
    49: 9,  # Crash 1 BOW -> CRASH
    55: 9,  # Crash 1 EDGE -> CRASH
    57: 9,  # Crash 2 BOW -> CRASH
    52: 9,  # Crash 2 EDGE -> CRASH
    # Toms (TD-17 has head and rim zones)
    48: 10,  # Tom 1 head -> TOM_HIGH
    50: 10,  # Tom 1 rim -> TOM_HIGH
    45: 11,  # Tom 2 head -> TOM_MID
    47: 11,  # Tom 2 rim -> TOM_MID
    43: 12,  # Tom 3 head -> TOM_LOW
    41: 12,  # Low Floor Tom -> TOM_LOW
    58: 12,  # Tom 3 rim -> TOM_LOW
    # Effects
    54: 13,  # Aux percussion -> EFFECTS
    56: 13,  # Cowbell -> EFFECTS
}


@dataclass
class SpectrogramConfig:
    """Spectrogram configuration matching the PyTorch version."""
    
    sample_rate: int = 44100
    n_fft: int = 2048
    hop_length: int = 441  # 10ms at 44.1kHz -> 100fps
    n_mels: int = 128
    fmin: float = 20.0
    fmax: float = 20000.0

    def time_to_frame(self, time_sec: float) -> int:
        """Convert time in seconds to frame index."""
        return int(time_sec * self.sample_rate / self.hop_length)

    def frame_to_time(self, frame: int) -> float:
        """Convert frame index to time in seconds."""
        return frame * self.hop_length / self.sample_rate


@dataclass
class DrumHit:
    """A single drum hit event."""
    time: float  # Time in seconds
    drum_class: int  # 0-13
    velocity: int  # 0-127


def load_midi_hits(midi_path: Path | str) -> list[DrumHit]:
    """Load drum hits from a MIDI file."""
    midi = mido.MidiFile(midi_path)
    hits = []
    current_time = 0.0

    for msg in midi:
        current_time += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            if msg.note in EXTENDED_MIDI_MAPPING:
                drum_class = EXTENDED_MIDI_MAPPING[msg.note]
                hit = DrumHit(
                    time=current_time,
                    drum_class=drum_class,
                    velocity=msg.velocity,
                )
                hits.append(hit)

    hits.sort(key=lambda h: h.time)
    return hits


def create_onset_target(
    hits: list[DrumHit],
    num_frames: int,
    config: SpectrogramConfig,
) -> tuple[mx.array, mx.array]:
    """Create onset and velocity targets from drum hits.
    
    Returns:
        Tuple of (onset_target, velocity_target) both shape (num_frames, num_classes)
    """
    onset_target = np.zeros((num_frames, NUM_CLASSES), dtype=np.float32)
    velocity_target = np.zeros((num_frames, NUM_CLASSES), dtype=np.float32)

    for hit in hits:
        frame = config.time_to_frame(hit.time)
        if 0 <= frame < num_frames:
            onset_target[frame, hit.drum_class] = 1.0
            velocity_target[frame, hit.drum_class] = hit.velocity / 127.0

    return mx.array(onset_target), mx.array(velocity_target)


def load_spectrogram_cached(audio_path: Path, config: SpectrogramConfig) -> mx.array:
    """Load spectrogram from cache or compute from audio.
    
    Checks for cached files in order: .spec.npy (MLX), .spec.pt (PyTorch)
    """
    npy_cache = audio_path.with_suffix(".spec.npy")
    pt_cache = audio_path.with_suffix(".spec.pt")
    
    if npy_cache.exists():
        # Load from numpy cache (fastest)
        spec_np = np.load(npy_cache)
        return mx.array(spec_np)
    
    if pt_cache.exists():
        # Load from PyTorch cache
        import torch
        spec_pt = torch.load(pt_cache, map_location="cpu", weights_only=True)
        spec_np = spec_pt.numpy()
        return mx.array(spec_np)

    # Would need to load audio and compute - for now raise
    raise FileNotFoundError(
        f"Spectrogram cache not found for {audio_path}. "
        "Expected .spec.npy or .spec.pt file."
    )


class EGMDDataset:
    """E-GMD dataset for MLX training."""

    def __init__(
        self,
        dataset_dir: Path | str,
        split: str | None = None,
        config: SpectrogramConfig | None = None,
        seq_length: int = 512,
        stride: int = 256,
    ):
        """Initialize dataset.
        
        Args:
            dataset_dir: Path to E-GMD dataset
            split: Data split ('train', 'validation', 'test')
            config: Spectrogram configuration
            seq_length: Number of frames per sample
            stride: Stride between samples
        """
        self.dataset_dir = Path(dataset_dir)
        self.config = config or SpectrogramConfig()
        self.seq_length = seq_length
        self.stride = stride

        # Load metadata
        self.examples = list(self._load_examples(split))
        
        # Pre-compute sample indices
        self.sample_indices: list[tuple[int, int]] = []
        self._compute_sample_indices()

    def _load_examples(self, split: str | None) -> Iterator[dict]:
        """Load examples from CSV."""
        csv_path = self.dataset_dir / "e-gmd-v1.0.0.csv"
        if not csv_path.exists():
            subdirs = list(self.dataset_dir.glob("e-gmd-*"))
            if subdirs:
                csv_path = subdirs[0] / "e-gmd-v1.0.0.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"E-GMD CSV not found at {csv_path}")

        base_dir = csv_path.parent

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split and row.get("split") != split:
                    continue

                midi_path = base_dir / row["midi_filename"]
                audio_path = base_dir / row["audio_filename"]
                
                # Check for .midi extension
                if not midi_path.exists():
                    midi_path = midi_path.with_suffix(".midi")
                
                # Check for spectrogram cache (.spec.pt or .spec.npy)
                pt_cache = audio_path.with_suffix(".spec.pt")
                npy_cache = audio_path.with_suffix(".spec.npy")
                has_spec = pt_cache.exists() or npy_cache.exists()

                if not midi_path.exists():
                    continue
                if not has_spec and not audio_path.exists():
                    continue

                yield {
                    "midi_path": midi_path,
                    "audio_path": audio_path,
                    "duration": float(row.get("duration", 0)),
                }

    def _compute_sample_indices(self):
        """Pre-compute indices for all training samples."""
        for ex_idx, example in enumerate(self.examples):
            num_frames = self.config.time_to_frame(example["duration"])
            for start in range(0, max(1, num_frames - self.seq_length), self.stride):
                self.sample_indices.append((ex_idx, start))

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> dict[str, mx.array]:
        """Get a training sample."""
        ex_idx, frame_start = self.sample_indices[idx]
        example = self.examples[ex_idx]

        # Load spectrogram
        full_spec = load_spectrogram_cached(example["audio_path"], self.config)

        # Extract window
        frame_end = frame_start + self.seq_length
        if frame_end > full_spec.shape[1]:
            # Pad if necessary
            spec = mx.zeros((self.config.n_mels, self.seq_length))
            valid_len = full_spec.shape[1] - frame_start
            spec = spec.at[:, :valid_len].add(full_spec[:, frame_start:])
        else:
            spec = full_spec[:, frame_start:frame_end]

        # Transpose to (time, freq) and add channel dim -> (time, freq, 1)
        spec = mx.transpose(spec, (1, 0))
        spec = mx.expand_dims(spec, axis=-1)

        # Load hits and create targets
        hits = load_midi_hits(example["midi_path"])

        # Filter hits to this window
        start_time = self.config.frame_to_time(frame_start)
        end_time = self.config.frame_to_time(frame_end)
        window_hits = [
            DrumHit(
                time=h.time - start_time,
                drum_class=h.drum_class,
                velocity=h.velocity,
            )
            for h in hits
            if start_time <= h.time < end_time
        ]

        onset_target, velocity_target = create_onset_target(
            window_hits, self.seq_length, self.config
        )

        return {
            "spectrogram": spec,
            "onset_target": onset_target,
            "velocity_target": velocity_target,
        }


def create_dataloader(
    dataset: EGMDDataset,
    batch_size: int = 32,
    shuffle: bool = True,
) -> Iterator[dict[str, mx.array]]:
    """Create a simple dataloader that yields batches.
    
    This is a generator that yields batches indefinitely.
    For finite iteration, track steps externally.
    """
    indices = list(range(len(dataset)))
    
    while True:
        if shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            
            # Collect batch
            specs = []
            onsets = []
            velocities = []
            
            for idx in batch_indices:
                sample = dataset[idx]
                specs.append(sample["spectrogram"])
                onsets.append(sample["onset_target"])
                velocities.append(sample["velocity_target"])
            
            yield {
                "spectrogram": mx.stack(specs),
                "onset_target": mx.stack(onsets),
                "velocity_target": mx.stack(velocities),
            }


def compute_class_weights(
    dataset: EGMDDataset,
    min_weight: float = 50.0,
    max_weight: float = 2000.0,
    max_samples: int = 1000,
) -> mx.array:
    """Compute per-class weights from dataset statistics.
    
    Returns:
        mx.array of shape (num_classes,) with per-class weights
    """
    class_counts = np.zeros(NUM_CLASSES)
    total_frames = 0

    # Sample subset of dataset
    sample_indices = np.random.choice(
        len(dataset),
        size=min(max_samples, len(dataset)),
        replace=False,
    )

    for idx in sample_indices:
        sample = dataset[int(idx)]
        onset_target = np.array(sample["onset_target"])
        class_counts += onset_target.sum(axis=0)
        total_frames += onset_target.shape[0]

    # Compute weights inversely proportional to frequency
    class_rates = class_counts / total_frames
    
    # Weight = 1 / rate, normalized so min weight = min_weight
    # Avoid division by zero
    class_rates = np.maximum(class_rates, 1e-8)
    weights = 1.0 / class_rates
    
    # Normalize
    weights = weights * (min_weight / weights.min())
    
    # Cap max weight
    weights = np.minimum(weights, max_weight)

    return mx.array(weights.astype(np.float32))
