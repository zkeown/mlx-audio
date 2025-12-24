"""
C++ extension loader with graceful fallback.

This module provides a single source of truth for C++ extension availability,
avoiding the need for repeated try/except blocks throughout the codebase.

Usage in other modules:
    from ._extension import HAS_CPP_EXT, _ext

    def some_function(...):
        if HAS_CPP_EXT and _ext is not None:
            return _ext.some_function(...)
        # Python fallback
        ...

The extension is optional - all functionality works without it, just potentially
slower for certain operations (overlap-add, signal framing).
"""

from __future__ import annotations

import logging
from typing import Any

# IMPORTANT: Import mlx.core BEFORE the C++ extension to ensure nanobind
# type casters are registered. The extension uses NB_DOMAIN mlx which requires
# MLX's Python module to be loaded first.
import mlx.core as _mx  # noqa: F401

_logger = logging.getLogger(__name__)

HAS_CPP_EXT: bool = False
_ext: Any | None = None

try:
    from . import _ext as _ext_module

    # Verify the extension actually works by calling a simple function.
    # This catches nanobind type caster issues that occur at runtime
    # (e.g., NB_DOMAIN mismatch between MLX wheel and our extension).
    _test_arr = _ext_module.generate_window("hann", 4, True)
    # If we get here, the extension works
    HAS_CPP_EXT = True
    _ext = _ext_module
except ImportError:
    # Extension not built - log info level since this is expected for pip install
    HAS_CPP_EXT = False
    _ext = None
    _logger.info(
        "C++ extensions not available. Using pure Python fallbacks. "
        "Performance may be reduced for STFT/overlap-add operations. "
        "To enable C++ acceleration, build from source with: pip install -e '.[dev]'"
    )
except TypeError as e:
    # Nanobind type caster issues (NB_DOMAIN mismatch) - this is unexpected
    HAS_CPP_EXT = False
    _ext = None
    _logger.warning(
        "C++ extension found but failed to load (nanobind type error: %s). "
        "This may indicate a version mismatch between MLX and mlx-audio. "
        "Using pure Python fallbacks instead.",
        e,
    )


def has_cpp_extensions() -> bool:
    """Check if C++ extensions are available.

    Returns:
        True if C++ extensions are loaded and functional, False otherwise.

    Example:
        >>> from mlx_audio.primitives import has_cpp_extensions
        >>> if has_cpp_extensions():
        ...     print("Using accelerated C++ primitives")
        ... else:
        ...     print("Using pure Python fallbacks")
    """
    return HAS_CPP_EXT


__all__ = ["_ext", "HAS_CPP_EXT", "has_cpp_extensions"]
