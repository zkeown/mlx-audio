"""Utility functions for mlx-audio.

This module provides common utilities used throughout the codebase,
including dependency management, weight conversion, and audio I/O.

Submodules:
    dependencies: Optional dependency import helpers
    conversion: Weight conversion utilities for PyTorch -> MLX
    audio_io: Audio file loading and saving utilities
"""

from __future__ import annotations

from mlx_audio.utils.dependencies import (
    require_dependency,
    require_torch,
    require_transformers,
    require_huggingface_hub,
    require_soundfile,
    require_librosa,
    require_scipy,
)

from mlx_audio.utils.conversion import (
    prepare_output_path,
    extract_state_dict,
    convert_conv1d_weight,
    convert_conv_transpose1d_weight,
    convert_weight,
    torch_to_numpy,
    save_mlx_weights,
    load_pytorch_checkpoint,
)

from mlx_audio.utils.audio_io import (
    load_audio_file,
    resample_audio,
    save_audio_file,
    normalize_audio_input,
)

__all__ = [
    # Dependencies
    "require_dependency",
    "require_torch",
    "require_transformers",
    "require_huggingface_hub",
    "require_soundfile",
    "require_librosa",
    "require_scipy",
    # Conversion
    "prepare_output_path",
    "extract_state_dict",
    "convert_conv1d_weight",
    "convert_conv_transpose1d_weight",
    "convert_weight",
    "torch_to_numpy",
    "save_mlx_weights",
    "load_pytorch_checkpoint",
    # Audio I/O
    "load_audio_file",
    "resample_audio",
    "save_audio_file",
    "normalize_audio_input",
]
