"""E-GMD dataset loader for MLX training.

Loads E-GMD drum transcription dataset and yields batches compatible
with MLX training.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import mido
import mlx.core as mx
import numpy as np

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


def load_midi_hits(midi_path: Path | str, use_cache: bool = True) -> list[DrumHit]:
    """Load drum hits from a MIDI file, using cache if available.

    Cache files are stored as .hits.npy alongside the MIDI file.
    """
    midi_path = Path(midi_path)
    cache_path = midi_path.with_suffix(".hits.npy")

    # Try to load from cache
    if use_cache and cache_path.exists():
        try:
            data = np.load(cache_path)
            return [
                DrumHit(time=float(row[0]), drum_class=int(row[1]), velocity=int(row[2]))
                for row in data
            ]
        except Exception:
            pass  # Fall back to parsing

    # Parse MIDI file
    midi = mido.MidiFile(midi_path)
    hits = []
    current_time = 0.0

    for msg in midi:
        current_time += msg.time
        if msg.type == "note_on" and msg.velocity > 0 and msg.note in EXTENDED_MIDI_MAPPING:
            drum_class = EXTENDED_MIDI_MAPPING[msg.note]
            hit = DrumHit(
                time=current_time,
                drum_class=drum_class,
                velocity=msg.velocity,
            )
            hits.append(hit)

    hits.sort(key=lambda h: h.time)

    # Save to cache
    if use_cache and hits:
        try:
            data = np.array(
                [(h.time, h.drum_class, h.velocity) for h in hits],
                dtype=np.float32,
            )
            np.save(cache_path, data)
        except Exception:
            pass  # Ignore cache write failures

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
        cache_spectrograms: bool = True,
        cache_midi: bool = True,
    ):
        """Initialize dataset.

        Args:
            dataset_dir: Path to E-GMD dataset
            split: Data split ('train', 'validation', 'test')
            config: Spectrogram configuration
            seq_length: Number of frames per sample
            stride: Stride between samples
            cache_spectrograms: Cache spectrograms in memory (uses more RAM but faster)
            cache_midi: Cache MIDI hits in memory (always recommended)
        """
        self.dataset_dir = Path(dataset_dir)
        self.config = config or SpectrogramConfig()
        self.seq_length = seq_length
        self.stride = stride
        self.cache_spectrograms = cache_spectrograms
        self.cache_midi = cache_midi

        # Caches
        self._spec_cache: dict[int, np.ndarray] = {}
        self._midi_cache: dict[int, list[DrumHit]] = {}

        # Load metadata
        self.examples = list(self._load_examples(split))

        # Pre-compute sample indices
        self.sample_indices: list[tuple[int, int]] = []
        self._compute_sample_indices()

        # Pre-cache MIDI files (they're small and parsing is slow)
        if self.cache_midi:
            self._preload_midi()

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

    def _preload_midi(self):
        """Pre-load all MIDI files into memory using parallel loading."""
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed

        num_examples = len(self.examples)
        print(f"  Pre-loading {num_examples:,} MIDI files...")

        # Use threads for I/O-bound MIDI parsing
        num_threads = min(32, os.cpu_count() or 8)

        def load_midi_item(item: tuple[int, dict]) -> tuple[int, list[DrumHit]]:
            ex_idx, example = item
            return ex_idx, load_midi_hits(example["midi_path"])

        items = list(enumerate(self.examples))
        loaded = 0

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(load_midi_item, item): item[0] for item in items}

            for future in as_completed(futures):
                ex_idx, hits = future.result()
                self._midi_cache[ex_idx] = hits
                loaded += 1

                # Progress update every 10%
                if loaded % (num_examples // 10 + 1) == 0:
                    pct = loaded * 100 // num_examples
                    print(f"    MIDI loading: {pct}% ({loaded:,}/{num_examples:,})")

    def _get_spectrogram(self, ex_idx: int) -> np.ndarray:
        """Get spectrogram, using cache if enabled."""
        if self.cache_spectrograms and ex_idx in self._spec_cache:
            return self._spec_cache[ex_idx]

        example = self.examples[ex_idx]
        spec = load_spectrogram_cached(example["audio_path"], self.config)
        spec_np = np.array(spec)  # Convert to numpy for slicing

        if self.cache_spectrograms:
            self._spec_cache[ex_idx] = spec_np

        return spec_np

    def _get_hits(self, ex_idx: int) -> list[DrumHit]:
        """Get MIDI hits, using cache if enabled."""
        if ex_idx in self._midi_cache:
            return self._midi_cache[ex_idx]

        example = self.examples[ex_idx]
        hits = load_midi_hits(example["midi_path"])

        if self.cache_midi:
            self._midi_cache[ex_idx] = hits

        return hits

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> dict[str, mx.array]:
        """Get a training sample."""
        ex_idx, frame_start = self.sample_indices[idx]

        # Load spectrogram (cached)
        full_spec = self._get_spectrogram(ex_idx)

        # Extract window using numpy (faster than mx slicing)
        frame_end = frame_start + self.seq_length
        if frame_end > full_spec.shape[1]:
            # Pad if necessary
            spec = np.zeros((self.config.n_mels, self.seq_length), dtype=np.float32)
            valid_len = full_spec.shape[1] - frame_start
            spec[:, :valid_len] = full_spec[:, frame_start:frame_start + valid_len]
        else:
            spec = full_spec[:, frame_start:frame_end]

        # Transpose to (time, freq) and add channel dim -> (time, freq, 1)
        spec = np.transpose(spec, (1, 0))
        spec = np.expand_dims(spec, axis=-1)

        # Load hits (cached) and create targets
        hits = self._get_hits(ex_idx)

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
            "spectrogram": mx.array(spec),
            "onset_target": onset_target,
            "velocity_target": velocity_target,
        }


def create_dataloader(
    dataset: EGMDDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    prefetch_factor: int = 2,
) -> Iterator[dict[str, mx.array]]:
    """Create a dataloader that yields batches.

    Args:
        dataset: The dataset to load from
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle indices each epoch
        num_workers: Number of background worker threads (0 = main thread only)
        prefetch_factor: Number of batches to prefetch per worker

    Returns:
        Iterator yielding batches indefinitely
    """
    if num_workers > 0:
        return _create_prefetching_dataloader(
            dataset, batch_size, shuffle, num_workers, prefetch_factor
        )
    else:
        return _create_simple_dataloader(dataset, batch_size, shuffle)


def _create_simple_dataloader(
    dataset: EGMDDataset,
    batch_size: int,
    shuffle: bool,
) -> Iterator[dict[str, mx.array]]:
    """Simple single-threaded dataloader."""
    indices = list(range(len(dataset)))

    while True:
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]

            # Collect batch using numpy arrays
            specs = []
            onsets = []
            velocities = []

            for idx in batch_indices:
                sample = dataset[idx]
                specs.append(np.array(sample["spectrogram"]))
                onsets.append(np.array(sample["onset_target"]))
                velocities.append(np.array(sample["velocity_target"]))

            yield {
                "spectrogram": mx.array(np.stack(specs)),
                "onset_target": mx.array(np.stack(onsets)),
                "velocity_target": mx.array(np.stack(velocities)),
            }


def _create_prefetching_dataloader(
    dataset: EGMDDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: int,
) -> Iterator[dict[str, mx.array]]:
    """Prefetching dataloader with background workers."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from queue import Queue

    # Queue for prefetched batches
    prefetch_queue: Queue = Queue(maxsize=num_workers * prefetch_factor)
    stop_event = threading.Event()

    def batch_loader(batch_indices: list[int]) -> dict[str, np.ndarray]:
        """Load a batch of samples (runs in worker thread)."""
        specs = []
        onsets = []
        velocities = []

        for idx in batch_indices:
            sample = dataset[idx]
            specs.append(np.array(sample["spectrogram"]))
            onsets.append(np.array(sample["onset_target"]))
            velocities.append(np.array(sample["velocity_target"]))

        return {
            "spectrogram": np.stack(specs),
            "onset_target": np.stack(onsets),
            "velocity_target": np.stack(velocities),
        }

    def prefetch_worker():
        """Worker that prefetches batches into the queue."""
        indices = list(range(len(dataset)))
        executor = ThreadPoolExecutor(max_workers=num_workers)

        while not stop_event.is_set():
            if shuffle:
                np.random.shuffle(indices)

            # Submit batches to thread pool
            futures = []
            for start in range(0, len(indices), batch_size):
                if stop_event.is_set():
                    break
                batch_indices = indices[start:start + batch_size]
                future = executor.submit(batch_loader, batch_indices)
                futures.append(future)

            # Collect results and put in queue
            for future in futures:
                if stop_event.is_set():
                    break
                try:
                    batch_np = future.result()
                    prefetch_queue.put(batch_np)
                except Exception as e:
                    print(f"Worker error: {e}")

        executor.shutdown(wait=False)

    # Start prefetch thread
    prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
    prefetch_thread.start()

    try:
        while True:
            batch_np = prefetch_queue.get()
            yield {
                "spectrogram": mx.array(batch_np["spectrogram"]),
                "onset_target": mx.array(batch_np["onset_target"]),
                "velocity_target": mx.array(batch_np["velocity_target"]),
            }
    finally:
        stop_event.set()


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
