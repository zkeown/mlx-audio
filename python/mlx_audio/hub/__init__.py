"""Model hub for mlx-audio.

Provides model registry, caching, and HuggingFace integration.
"""

from mlx_audio.hub.cache import (
    ModelCache,
    clear_cache,
    configure_cache,
    get_cache,
    preload_models,
)
from mlx_audio.hub.licenses import (
    LicenseInfo,
    LicenseType,
    NonCommercialModelWarning,
    get_license_info,
    is_commercial_safe,
    list_commercial_safe_models,
    list_non_commercial_models,
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
    # Licenses
    "LicenseInfo",
    "LicenseType",
    "NonCommercialModelWarning",
    "get_license_info",
    "is_commercial_safe",
    "list_commercial_safe_models",
    "list_non_commercial_models",
    # Registry
    "ModelRegistry",
    "ModelSpec",
    "TaskType",
    "register_model",
]
