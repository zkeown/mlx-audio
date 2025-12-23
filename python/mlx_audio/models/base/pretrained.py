"""Pretrained model loading mixin.

Provides the from_pretrained() class method for loading models
from local paths or cached checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from mlx_audio.models.base.config import ModelConfig


class PretrainedMixin:
    """Mixin providing from_pretrained functionality.

    Classes using this mixin must:
    1. Define a `config_class` attribute pointing to their config class
    2. Accept a config as the first argument to __init__
    3. Have a `load_weights` method for loading saved weights

    Example:
        >>> class MyModel(nn.Module, PretrainedMixin):
        ...     config_class = MyModelConfig
        ...
        ...     def __init__(self, config: MyModelConfig):
        ...         super().__init__()
        ...         self.config = config
        ...         # ... build layers ...
        ...
        >>> model = MyModel.from_pretrained("path/to/model")
    """

    # Must be set by subclass
    config_class: type[ModelConfig]

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        **config_overrides: Any,
    ) -> Self:
        """Load a pretrained model from a directory.

        The directory should contain:
        - config.json: Model configuration
        - model.safetensors or weights.npz: Model weights

        Args:
            path: Path to model directory
            **config_overrides: Override specific config values

        Returns:
            Loaded model instance

        Example:
            >>> model = Whisper.from_pretrained("~/.cache/mlx_audio/whisper-large")
            >>> model = Whisper.from_pretrained("path/to/model", n_text_layer=4)
        """
        path = Path(path).expanduser()

        # Load or create config
        config_path = path / "config.json"
        if config_path.exists():
            config = cls.config_class.from_json(config_path)
        else:
            # Use default config if no config.json
            config = cls.config_class()

        # Apply any config overrides
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Create model instance
        model = cls(config)

        # Load weights
        model._load_weights_from_path(path)

        return model

    def _load_weights_from_path(self, path: Path) -> None:
        """Load weights from a model directory.

        Tries safetensors first, then falls back to npz format.

        Args:
            path: Path to model directory
        """
        # Try safetensors format first
        weights_path = path / "model.safetensors"
        if weights_path.exists():
            self.load_weights(str(weights_path))
            return

        # Fall back to npz format
        npz_path = path / "weights.npz"
        if npz_path.exists():
            self.load_weights(str(npz_path))
            return

        # Check for any safetensors file
        safetensors_files = list(path.glob("*.safetensors"))
        if safetensors_files:
            self.load_weights(str(safetensors_files[0]))
            return

        # No weights found - this may be intentional for a freshly initialized model
        # Don't raise an error, just return

    def save_pretrained(self, path: str | Path) -> None:
        """Save model to a directory.

        Saves both config.json and model.safetensors.

        Args:
            path: Directory to save model to
        """
        import mlx.core as mx

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        if hasattr(self, "config"):
            self.config.to_json(path / "config.json")

        # Save weights
        weights = dict(self.parameters())
        mx.save_safetensors(str(path / "model.safetensors"), weights)


__all__ = ["PretrainedMixin"]
