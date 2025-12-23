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

from typing import Any

# IMPORTANT: Import mlx.core BEFORE the C++ extension to ensure nanobind
# type casters are registered. The extension uses NB_DOMAIN mlx which requires
# MLX's Python module to be loaded first.
import mlx.core as _mx  # noqa: F401

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
except (ImportError, TypeError):
    # ImportError: extension not built
    # TypeError: nanobind type caster issues (NB_DOMAIN mismatch)
    HAS_CPP_EXT = False
    _ext = None

__all__ = ["_ext", "HAS_CPP_EXT"]
