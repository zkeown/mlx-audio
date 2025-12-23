"""Dependency management utilities.

Provides helpers for importing optional dependencies with
user-friendly error messages.
"""

from __future__ import annotations

import importlib
from typing import Any


def require_dependency(
    package: str,
    install_name: str | None = None,
    purpose: str | None = None,
) -> Any:
    """Import a package, raising a helpful error if missing.

    Args:
        package: The package name to import (e.g., "torch")
        install_name: The pip install name if different from package name
        purpose: Optional description of why this dependency is needed

    Returns:
        The imported module

    Raises:
        ImportError: If the package is not installed, with installation instructions

    Example:
        >>> torch = require_dependency("torch")
        >>> sf = require_dependency("soundfile", purpose="audio file loading")
    """
    install_name = install_name or package
    try:
        return importlib.import_module(package)
    except ImportError:
        purpose_msg = f" for {purpose}" if purpose else ""
        raise ImportError(
            f"{package} is required{purpose_msg}. "
            f"Install with: pip install {install_name}"
        )


def require_torch() -> Any:
    """Import PyTorch, raising helpful error if missing.

    Returns:
        The torch module

    Raises:
        ImportError: If PyTorch is not installed
    """
    return require_dependency("torch", purpose="weight conversion")


def require_transformers() -> Any:
    """Import HuggingFace transformers, raising helpful error if missing.

    Returns:
        The transformers module

    Raises:
        ImportError: If transformers is not installed
    """
    return require_dependency("transformers", purpose="model conversion")


def require_huggingface_hub() -> Any:
    """Import huggingface_hub, raising helpful error if missing.

    Returns:
        The huggingface_hub module

    Raises:
        ImportError: If huggingface_hub is not installed
    """
    return require_dependency(
        "huggingface_hub",
        install_name="huggingface-hub",
        purpose="model downloading",
    )


def require_soundfile() -> Any:
    """Import soundfile, raising helpful error if missing.

    Returns:
        The soundfile module

    Raises:
        ImportError: If soundfile is not installed
    """
    return require_dependency("soundfile", purpose="audio file I/O")


def require_librosa() -> Any:
    """Import librosa, raising helpful error if missing.

    Returns:
        The librosa module

    Raises:
        ImportError: If librosa is not installed
    """
    return require_dependency("librosa", purpose="audio processing")


def require_scipy() -> Any:
    """Import scipy, raising helpful error if missing.

    Returns:
        The scipy module

    Raises:
        ImportError: If scipy is not installed
    """
    return require_dependency("scipy", purpose="signal processing")


__all__ = [
    "require_dependency",
    "require_torch",
    "require_transformers",
    "require_huggingface_hub",
    "require_soundfile",
    "require_librosa",
    "require_scipy",
]
