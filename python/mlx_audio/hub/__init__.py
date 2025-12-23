"""Model hub for mlx-audio.

Provides model registry, caching, and HuggingFace integration.
"""

from mlx_audio.hub.cache import (
    ModelCache,
    get_cache,
    configure_cache,
    clear_cache,
    preload_models,
)
from mlx_audio.hub.registry import (
    ModelRegistry,
    ModelSpec,
    TaskType,
    register_model,
)

__all__ = [
    # Cache
    "ModelCache",
    "get_cache",
    "configure_cache",
    "clear_cache",
    "preload_models",
    # Registry
    "ModelRegistry",
    "ModelSpec",
    "TaskType",
    "register_model",
]
