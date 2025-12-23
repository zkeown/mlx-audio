"""Base configuration class for mlx-audio models.

Provides common serialization and factory methods shared by
all model configuration classes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, ClassVar, Self


class ModelConfig:
    """Base class for model configurations.

    Provides common serialization methods (from_dict, to_dict, from_json, to_json)
    that are shared across all model configuration classes.

    Subclasses should be decorated with @dataclass and define their
    configuration fields as class attributes.

    Example:
        >>> @dataclass
        ... class MyModelConfig(ModelConfig):
        ...     hidden_size: int = 256
        ...     num_layers: int = 4
        ...
        >>> config = MyModelConfig.from_dict({"hidden_size": 512})
        >>> config.to_dict()
        {'hidden_size': 512, 'num_layers': 4}
    """

    # Subclasses can define a mapping of name -> factory method
    # for use with from_name()
    _config_registry: ClassVar[dict[str, str]] = {}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Create a config instance from a dictionary.

        Only keys that correspond to valid fields are used.
        Extra keys are silently ignored.

        Args:
            d: Dictionary of configuration values

        Returns:
            Config instance with values from dictionary
        """
        # Get valid field names from dataclass
        try:
            valid_fields = {f.name for f in fields(cls)}
        except TypeError:
            # Not a dataclass, fall back to __init__ parameters
            import inspect

            sig = inspect.signature(cls.__init__)
            valid_fields = set(sig.parameters.keys()) - {"self"}

        return cls(**{k: v for k, v in d.items() if k in valid_fields})

    def to_dict(self) -> dict[str, Any]:
        """Convert config to a dictionary.

        Returns:
            Dictionary representation of the config
        """
        try:
            return asdict(self)
        except TypeError:
            # Not a dataclass, fall back to __dict__
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_json(cls, path: str | Path) -> Self:
        """Load config from a JSON file.

        Args:
            path: Path to JSON file

        Returns:
            Config instance loaded from file
        """
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_json(self, path: str | Path, indent: int = 2) -> None:
        """Save config to a JSON file.

        Args:
            path: Path to save JSON file
            indent: JSON indentation level (default: 2)
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def from_name(cls, name: str) -> Self:
        """Create config from a preset name.

        Subclasses should define a _config_registry mapping names
        to factory method names, or override this method.

        Args:
            name: Model name or preset identifier

        Returns:
            Config instance for the specified preset

        Raises:
            ValueError: If name is not recognized
        """
        # Normalize name
        normalized = name.lower().replace("-", "_").replace(".", "_")

        # Check for registered factory methods
        if hasattr(cls, "_config_registry") and normalized in cls._config_registry:
            method_name = cls._config_registry[normalized]
            return getattr(cls, method_name)()

        # Try direct method lookup
        if hasattr(cls, normalized):
            method = getattr(cls, normalized)
            if callable(method):
                return method()

        # Build list of available configs
        available = []
        if hasattr(cls, "_config_registry"):
            available.extend(cls._config_registry.keys())

        # Also check for classmethod factory methods
        for attr_name in dir(cls):
            if attr_name.startswith("_"):
                continue
            attr = getattr(cls, attr_name)
            if callable(attr) and hasattr(attr, "__self__"):
                available.append(attr_name)

        raise ValueError(
            f"Unknown model name: {name!r}. "
            f"Available: {', '.join(sorted(set(available)))}"
        )


__all__ = ["ModelConfig"]
