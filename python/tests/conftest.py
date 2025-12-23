"""Pytest configuration and fixtures for mlx-audio tests."""

import numpy as np
import pytest

try:
    import mlx.core as mx
except ImportError:
    mx = None

_TEST_SEED = 42


@pytest.fixture
def fixed_seed() -> int:
    """Fix MLX and NumPy random seeds."""
    if mx is not None:
        mx.random.seed(_TEST_SEED)
    np.random.seed(_TEST_SEED)
    return _TEST_SEED


@pytest.fixture
def random_audio():
    """Random audio signal for testing."""
    rng = np.random.default_rng(_TEST_SEED)
    return rng.standard_normal(22050).astype(np.float32)


@pytest.fixture
def stereo_audio():
    """Random stereo audio for testing."""
    rng = np.random.default_rng(_TEST_SEED)
    return rng.standard_normal((2, 44100)).astype(np.float32)
