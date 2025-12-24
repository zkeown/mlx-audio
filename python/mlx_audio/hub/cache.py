"""Model caching for mlx-audio.

Provides LRU caching for loaded models with HuggingFace hub integration.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Type


@dataclass
class ModelCache:
    """Model caching with MLX-optimized storage.

    Features:
    - Leverages HuggingFace hub's revision-based caching
    - Maintains in-memory LRU cache of loaded models (O(1) eviction)
    - Supports explicit cache management (clear, preload)
    - Thread-safe model loading

    Attributes:
        cache_dir: Directory for cached model files
        max_memory_models: Maximum models to keep loaded in memory
    """

    cache_dir: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "mlx_audio"
    )
    max_memory_models: int = 3

    # OrderedDict provides O(1) LRU operations via move_to_end() and popitem()
    _loaded_models: OrderedDict[str, Any] = field(
        default_factory=OrderedDict, repr=False
    )
    # Global lock for cache structure modifications
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    # Per-model locks enable concurrent loading of different models
    _model_locks: dict[str, threading.Lock] = field(
        default_factory=dict, repr=False
    )
    _model_locks_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False
    )

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_lock(self, cache_key: str) -> threading.Lock:
        """Get or create a lock for a specific model (thread-safe)."""
        with self._model_locks_lock:
            if cache_key not in self._model_locks:
                self._model_locks[cache_key] = threading.Lock()
            return self._model_locks[cache_key]

    def get_model_path(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        force_download: bool = False,
    ) -> Path:
        """Get local path to model files, downloading if necessary.

        Args:
            repo_id: HuggingFace repository ID or local path
            revision: Specific revision (commit, branch, tag)
            force_download: Re-download even if cached

        Returns:
            Path to local model directory
        """
        # Check if it's a local path
        local_path = Path(repo_id)
        if local_path.exists():
            return local_path

        # Check if it's a short model name in our local cache
        if "/" not in repo_id:
            cached_path = self.cache_dir / "models" / repo_id
            if cached_path.exists():
                return cached_path

        # Check for short model names (htdemucs_ft, etc.)
        if "/" not in repo_id:
            # Try to resolve from registry
            from mlx_audio.hub.registry import ModelRegistry, TaskType

            registry = ModelRegistry.get()
            spec = registry.get_spec(repo_id)
            if spec:
                repo_id = spec.default_repo

        try:
            from huggingface_hub import snapshot_download

            return Path(
                snapshot_download(
                    repo_id,
                    revision=revision,
                    cache_dir=self.cache_dir / "hub",
                    force_download=force_download,
                    local_files_only=False,
                )
            )
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for model downloading. "
                "Install with: pip install huggingface-hub"
            )

    def get_model(
        self,
        model_id: str,
        model_class: Type,
        *,
        revision: str | None = None,
        force_reload: bool = False,
        **load_kwargs,
    ) -> Any:
        """Get a loaded model instance, using cache when possible.

        Implements LRU eviction for memory management.

        Args:
            model_id: Model name or HuggingFace repo ID
            model_class: Class to instantiate
            revision: Specific model revision
            force_reload: Force reload even if cached
            **load_kwargs: Additional arguments for model loading

        Returns:
            Loaded model instance
        """
        cache_key = self._make_cache_key(model_id, revision, load_kwargs)

        # Quick check with global lock (fast path for cached models)
        with self._lock:
            if not force_reload and cache_key in self._loaded_models:
                # Move to end (most recently used) - O(1) with OrderedDict
                self._loaded_models.move_to_end(cache_key)
                return self._loaded_models[cache_key]

        # Per-model lock enables concurrent loading of different models
        model_lock = self._get_model_lock(cache_key)
        with model_lock:
            # Double-check after acquiring lock (another thread may have loaded)
            with self._lock:
                if not force_reload and cache_key in self._loaded_models:
                    self._loaded_models.move_to_end(cache_key)
                    return self._loaded_models[cache_key]

            # Check license and warn if non-commercial
            from mlx_audio.hub.licenses import check_license_and_warn

            check_license_and_warn(model_id)

            # Get model path (download if needed) - concurrent for different models
            model_path = self.get_model_path(model_id, revision=revision)

            # Load model - can run concurrently for different models
            if hasattr(model_class, "from_pretrained"):
                model = model_class.from_pretrained(model_path, **load_kwargs)
            else:
                model = model_class(model_path, **load_kwargs)

            # Add to cache with LRU eviction (requires global lock)
            with self._lock:
                # Check again in case it was added while we were loading
                if cache_key in self._loaded_models:
                    # Another thread beat us, use their version
                    self._loaded_models.move_to_end(cache_key)
                    return self._loaded_models[cache_key]

                # Evict oldest if at capacity - O(1) with OrderedDict
                if len(self._loaded_models) >= self.max_memory_models:
                    self._loaded_models.popitem(last=False)

                self._loaded_models[cache_key] = model

            return model

    def preload(self, repo_ids: list[str]) -> None:
        """Pre-download models without loading into memory.

        Useful for offline preparation or CI/CD pipelines.

        Args:
            repo_ids: List of repository IDs to download
        """
        for repo_id in repo_ids:
            self.get_model_path(repo_id)

    def clear_memory_cache(self) -> None:
        """Unload all models from memory (keeps disk cache)."""
        with self._lock:
            self._loaded_models.clear()

    def clear_disk_cache(self, repo_id: str | None = None) -> None:
        """Remove cached model files from disk.

        Args:
            repo_id: Specific repo to clear, or None for all
        """
        try:
            from huggingface_hub import scan_cache_dir

            cache_info = scan_cache_dir(self.cache_dir / "hub")

            for repo in cache_info.repos:
                if repo_id is None or repo.repo_id == repo_id:
                    for revision in repo.revisions:
                        revision.delete()
        except ImportError:
            # No huggingface_hub, just clear the directory
            import shutil

            if repo_id is None:
                shutil.rmtree(self.cache_dir / "hub", ignore_errors=True)

    def _make_cache_key(
        self,
        repo_id: str,
        revision: str | None,
        kwargs: dict,
    ) -> str:
        """Create a unique cache key for a model configuration."""
        key_data = {
            "repo": repo_id,
            "revision": revision or "main",
            **kwargs,
        }
        return hashlib.sha256(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()[:16]


# Global cache instance
_cache: ModelCache | None = None


def get_cache() -> ModelCache:
    """Get the global model cache instance."""
    global _cache
    if _cache is None:
        _cache = ModelCache()
    return _cache


def configure_cache(
    cache_dir: Path | str | None = None,
    max_memory_models: int | None = None,
) -> None:
    """Configure the global model cache.

    Args:
        cache_dir: Custom cache directory
        max_memory_models: Maximum models to keep in memory

    Example:
        mlx_audio.configure_cache(
            cache_dir=Path("/custom/cache"),
            max_memory_models=5
        )
    """
    global _cache
    kwargs = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = Path(cache_dir)
    if max_memory_models is not None:
        kwargs["max_memory_models"] = max_memory_models
    _cache = ModelCache(**kwargs)


def clear_cache(memory: bool = True, disk: bool = False) -> None:
    """Clear the model cache.

    Args:
        memory: Clear in-memory cache
        disk: Clear disk cache
    """
    cache = get_cache()
    if memory:
        cache.clear_memory_cache()
    if disk:
        cache.clear_disk_cache()


def preload_models(repo_ids: list[str]) -> None:
    """Pre-download models for offline use.

    Args:
        repo_ids: List of repository IDs to download
    """
    get_cache().preload(repo_ids)
