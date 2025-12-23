"""Pytest fixtures for classification tests."""

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
def random_embedding():
    """Random 512-dim embedding (CLAP embedding size)."""
    if mx is None:
        pytest.skip("MLX not available")
    mx.random.seed(_TEST_SEED)
    return mx.random.normal((512,))


@pytest.fixture
def batch_embeddings():
    """Batch of random embeddings [B, 512]."""
    if mx is None:
        pytest.skip("MLX not available")
    mx.random.seed(_TEST_SEED)
    return mx.random.normal((8, 512))


@pytest.fixture
def sample_labels():
    """Sample label names for testing."""
    return ["dog", "cat", "bird", "car", "train"]


@pytest.fixture
def multilabel_sample():
    """Sample with multiple labels."""
    return ["dog", "bird"]  # Two active labels
