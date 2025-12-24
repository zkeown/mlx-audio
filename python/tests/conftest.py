"""Pytest configuration and fixtures for mlx-audio tests."""

import sys
import numpy as np
import pytest

# Compatibility shim for deepfilternet with newer torchaudio versions
# torchaudio 2.x removed the backend.common module that deepfilternet expects
try:
    import torchaudio
    if not hasattr(torchaudio, 'backend'):
        from types import ModuleType

        # Create mock AudioMetaData class
        class AudioMetaData:
            def __init__(self, sample_rate=None, num_frames=None, num_channels=None,
                         bits_per_sample=None, encoding=None):
                self.sample_rate = sample_rate
                self.num_frames = num_frames
                self.num_channels = num_channels
                self.bits_per_sample = bits_per_sample
                self.encoding = encoding

        # Create mock modules
        backend_common = ModuleType('torchaudio.backend.common')
        backend_common.AudioMetaData = AudioMetaData

        backend = ModuleType('torchaudio.backend')
        backend.common = backend_common

        # Register in sys.modules and as attribute
        sys.modules['torchaudio.backend'] = backend
        sys.modules['torchaudio.backend.common'] = backend_common
        torchaudio.backend = backend
except ImportError:
    pass

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
