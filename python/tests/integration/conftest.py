"""Pytest fixtures for integration tests with real audio datasets."""

import os
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import soundfile as sf


# =============================================================================
# Dataset Root Fixtures
# =============================================================================


@pytest.fixture
def musdb18_root() -> Path:
    """Get MUSDB18-HQ dataset root from environment or skip.

    Set MUSDB18_ROOT environment variable to the path containing train/ and test/
    Example: MUSDB18_ROOT=/Users/zakkeown/datasets/MUSDB18HQ
    """
    root = os.environ.get("MUSDB18_ROOT")
    if root is None:
        pytest.skip("MUSDB18_ROOT not set - set to path containing train/ and test/")
    root_path = Path(root)
    if not root_path.exists():
        pytest.skip(f"MUSDB18_ROOT {root} does not exist")
    if not (root_path / "test").exists():
        pytest.skip(f"MUSDB18_ROOT {root} does not contain test/ directory")
    return root_path


@pytest.fixture
def esc50_root() -> Path:
    """Get ESC-50 dataset root from environment or skip.

    Set ESC50_ROOT environment variable to the path containing audio/ and meta/
    Example: ESC50_ROOT=/Users/zakkeown/datasets/ESC-50
    """
    root = os.environ.get("ESC50_ROOT")
    if root is None:
        pytest.skip("ESC50_ROOT not set - set to ESC-50 root with audio/ and meta/")
    root_path = Path(root)
    if not root_path.exists():
        pytest.skip(f"ESC50_ROOT {root} does not exist")
    if not (root_path / "audio").exists():
        pytest.skip(f"ESC50_ROOT {root} does not contain audio/ directory")
    return root_path


# =============================================================================
# MUSDB18 Track Fixtures
# =============================================================================


def get_musdb18_test_tracks(root: Path) -> list[str]:
    """Get list of track names in MUSDB18-HQ test set."""
    test_dir = root / "test"
    if not test_dir.exists():
        return []
    return sorted([d.name for d in test_dir.iterdir() if d.is_dir()])


@pytest.fixture
def all_test_tracks(musdb18_root: Path) -> list[str]:
    """Get all 50 test tracks from MUSDB18-HQ."""
    tracks = get_musdb18_test_tracks(musdb18_root)
    if len(tracks) == 0:
        pytest.skip("No tracks found in MUSDB18_ROOT/test/")
    return tracks


@pytest.fixture
def quick_tracks(musdb18_root: Path) -> list[str]:
    """Get 5 representative tracks for quick smoke testing (~5 min).

    Selected for variety: different durations, genres, complexity.
    """
    all_tracks = get_musdb18_test_tracks(musdb18_root)
    if len(all_tracks) == 0:
        pytest.skip("No tracks found in MUSDB18_ROOT/test/")

    # Select 5 tracks - use first 5 if not enough variety
    # These are typically good test candidates
    preferred = [
        "Al James - Schoolboy Facination",
        "AM Contra - Heart Peripheral",
        "Angels In Amplifiers - I'm Alright",
        "Arise - Run Run Run",
        "BKS - Bulldozer",
    ]

    selected = []
    for track in preferred:
        if track in all_tracks:
            selected.append(track)
        if len(selected) >= 5:
            break

    # Fill with remaining tracks if needed
    for track in all_tracks:
        if track not in selected:
            selected.append(track)
        if len(selected) >= 5:
            break

    return selected[:5]


# =============================================================================
# Track Loading Utilities
# =============================================================================


def load_musdb18_track(
    root: Path,
    track_name: str,
    split: str = "test",
    duration: float | None = None,
) -> dict[str, np.ndarray]:
    """Load a MUSDB18-HQ track with all stems.

    Args:
        root: MUSDB18-HQ root directory
        track_name: Name of track (e.g., "Al James - Schoolboy Facination")
        split: "train" or "test"
        duration: Optional duration in seconds to load (from start)

    Returns:
        Dictionary with keys: mixture, drums, bass, other, vocals
        Each value is np.ndarray of shape [channels, samples] at 44100 Hz
    """
    track_dir = root / split / track_name

    stems = {}
    stem_names = ["mixture", "drums", "bass", "other", "vocals"]

    for stem in stem_names:
        wav_path = track_dir / f"{stem}.wav"
        if not wav_path.exists():
            raise FileNotFoundError(f"Stem not found: {wav_path}")

        audio, sr = sf.read(wav_path, dtype="float32")
        assert sr == 44100, f"Expected 44100 Hz, got {sr}"

        # Transpose to [channels, samples]
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        else:
            audio = audio.T

        # Optionally truncate to duration
        if duration is not None:
            max_samples = int(duration * sr)
            audio = audio[:, :max_samples]

        stems[stem] = audio

    return stems


@pytest.fixture
def load_track(musdb18_root: Path):
    """Fixture that returns a track loading function."""

    def _load(track_name: str, duration: float | None = None) -> dict[str, np.ndarray]:
        return load_musdb18_track(musdb18_root, track_name, duration=duration)

    return _load


# =============================================================================
# ESC-50 Fixtures
# =============================================================================


def load_esc50_metadata(root: Path) -> list[dict]:
    """Load ESC-50 metadata from CSV."""
    import csv

    meta_path = root / "meta" / "esc50.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"ESC-50 metadata not found: {meta_path}")

    with open(meta_path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_esc50_classes() -> list[str]:
    """Get the 50 ESC-50 class names in order."""
    return [
        "dog", "rooster", "pig", "cow", "frog",
        "cat", "hen", "insects", "sheep", "crow",
        "rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds",
        "water_drops", "wind", "pouring_water", "toilet_flush", "thunderstorm",
        "crying_baby", "sneezing", "clapping", "breathing", "coughing",
        "footsteps", "laughing", "brushing_teeth", "snoring", "drinking_sipping",
        "door_wood_knock", "mouse_click", "keyboard_typing", "door_wood_creaks", "can_opening",
        "washing_machine", "vacuum_cleaner", "clock_alarm", "clock_tick", "glass_breaking",
        "helicopter", "chainsaw", "siren", "car_horn", "engine",
        "train", "church_bells", "airplane", "fireworks", "hand_saw",
    ]


@pytest.fixture
def esc50_metadata(esc50_root: Path) -> list[dict]:
    """Load ESC-50 metadata."""
    return load_esc50_metadata(esc50_root)


@pytest.fixture
def esc50_classes() -> list[str]:
    """Get ESC-50 class names."""
    return get_esc50_classes()


@pytest.fixture
def esc50_fold(esc50_root: Path, esc50_metadata: list[dict]):
    """Get samples for a specific fold."""

    def _get_fold(fold: int) -> list[dict]:
        return [m for m in esc50_metadata if int(m["fold"]) == fold]

    return _get_fold


def load_esc50_audio(root: Path, filename: str) -> tuple[np.ndarray, int]:
    """Load an ESC-50 audio file.

    Returns:
        Tuple of (audio array [samples], sample_rate)
    """
    audio_path = root / "audio" / filename
    audio, sr = sf.read(audio_path, dtype="float32")
    return audio, sr


@pytest.fixture
def load_esc50_clip(esc50_root: Path):
    """Fixture that returns an ESC-50 audio loading function."""

    def _load(filename: str) -> tuple[np.ndarray, int]:
        return load_esc50_audio(esc50_root, filename)

    return _load


# =============================================================================
# Model Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def htdemucs_model():
    """Load HTDemucs model (cached per module)."""
    from mlx_audio.models.demucs import HTDemucs

    model = HTDemucs.from_pretrained("htdemucs_ft")
    return model


@pytest.fixture(scope="module")
def htdemucs_bag():
    """Load HTDemucs BagOfModels ensemble (cached per module).

    Note: BagOfModels requires special pretrained weights with multiple models.
    Skip if not available.
    """
    pytest.skip("BagOfModels requires special multi-model weights not in htdemucs_ft")


@pytest.fixture(scope="module")
def clap_model():
    """Load CLAP model (cached per module)."""
    from mlx_audio.models.clap import CLAP

    model = CLAP.from_pretrained("clap-htsat-fused")
    return model


# =============================================================================
# Quality Metrics Utilities
# =============================================================================


def compute_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Compute Signal-to-Distortion Ratio (SDR) in dB.

    Uses mir_eval for professional BSSEval computation.

    Args:
        reference: Ground truth signal [channels, samples]
        estimate: Estimated signal [channels, samples]

    Returns:
        SDR in dB (higher is better)
    """
    try:
        from mir_eval.separation import bss_eval_sources
    except ImportError:
        pytest.skip("mir_eval not installed - pip install mir_eval")

    # mir_eval expects [sources, samples] - average over channels
    ref = reference.mean(axis=0, keepdims=True)
    est = estimate.mean(axis=0, keepdims=True)

    sdr, sir, sar, _ = bss_eval_sources(ref, est, compute_permutation=False)
    return float(sdr[0])


def compute_all_metrics(
    reference: np.ndarray, estimate: np.ndarray
) -> dict[str, float]:
    """Compute all separation metrics.

    Returns:
        Dictionary with SDR, SIR, SAR in dB
    """
    try:
        from mir_eval.separation import bss_eval_sources
    except ImportError:
        pytest.skip("mir_eval not installed - pip install mir_eval")

    ref = reference.mean(axis=0, keepdims=True)
    est = estimate.mean(axis=0, keepdims=True)

    sdr, sir, sar, _ = bss_eval_sources(ref, est, compute_permutation=False)
    return {
        "sdr": float(sdr[0]),
        "sir": float(sir[0]),
        "sar": float(sar[0]),
    }


@pytest.fixture
def compute_metrics():
    """Fixture that returns metric computation function."""
    return compute_all_metrics
