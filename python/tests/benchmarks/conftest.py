"""Pytest fixtures for benchmark tests."""

import os
from pathlib import Path

import pytest


@pytest.fixture
def esc50_root() -> Path | None:
    """Get ESC-50 dataset root from environment or skip.

    Set ESC50_ROOT environment variable to the path containing ESC-50-master/
    """
    root = os.environ.get("ESC50_ROOT")
    if root is None:
        pytest.skip("ESC50_ROOT not set")
    root_path = Path(root)
    if not root_path.exists():
        pytest.skip(f"ESC50_ROOT {root} does not exist")
    return root_path


@pytest.fixture
def fsd50k_root() -> Path | None:
    """Get FSD50K dataset root from environment or skip.

    Set FSD50K_ROOT environment variable to the path containing FSD50K.*/
    """
    root = os.environ.get("FSD50K_ROOT")
    if root is None:
        pytest.skip("FSD50K_ROOT not set")
    root_path = Path(root)
    if not root_path.exists():
        pytest.skip(f"FSD50K_ROOT {root} does not exist")
    return root_path


@pytest.fixture
def audioset_root() -> Path | None:
    """Get AudioSet dataset root from environment or skip.

    Set AUDIOSET_ROOT environment variable to the path containing audio/ and metadata/
    """
    root = os.environ.get("AUDIOSET_ROOT")
    if root is None:
        pytest.skip("AUDIOSET_ROOT not set")
    root_path = Path(root)
    if not root_path.exists():
        pytest.skip(f"AUDIOSET_ROOT {root} does not exist")
    return root_path
