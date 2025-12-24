"""Base class for PyTorch to MLX weight conversion.

This module provides a common infrastructure for converting PyTorch model
weights to MLX safetensors format. Each model's convert.py should subclass
WeightConverter and implement the model-specific key mapping and weight
transformation logic.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np

from mlx_audio.exceptions import WeightConversionError


class WeightConverter(ABC):
    """Base class for PyTorch to MLX weight conversion.

    Subclasses must implement:
    - map_key(): Map PyTorch parameter names to MLX names
    - transform_weight(): Transform weight arrays (e.g., transpose conv kernels)

    The convert() method handles the common workflow:
    1. Load PyTorch checkpoint
    2. Extract state dict (handles various checkpoint formats)
    3. Loop over parameters, mapping keys and transforming weights
    4. Save as safetensors with optional config.json

    Example:
        >>> class HTDemucsConverter(WeightConverter):
        ...     def map_key(self, pt_key: str) -> str | None:
        ...         # Return None to skip parameter
        ...         if "num_batches_tracked" in pt_key:
        ...             return None
        ...         return pt_key.replace("encoder", "enc")
        ...
        ...     def transform_weight(self, key: str, arr: np.ndarray) -> np.ndarray:
        ...         if "conv" in key and "weight" in key and arr.ndim == 3:
        ...             return arr.transpose(2, 1, 0)  # PyTorch -> MLX conv format
        ...         return arr
        ...
        >>> converter = HTDemucsConverter()
        >>> converter.convert("htdemucs.th", "htdemucs/model.safetensors")
    """

    # Override in subclasses if needed
    model_name: str = "model"

    def __init__(self, verbose: bool = True):
        """Initialize converter.

        Args:
            verbose: Print progress messages during conversion
        """
        self.verbose = verbose

    @abstractmethod
    def map_key(self, pt_key: str) -> str | None:
        """Map PyTorch parameter key to MLX key.

        Args:
            pt_key: PyTorch parameter name (e.g., "encoder.0.conv.weight")

        Returns:
            MLX parameter name, or None to skip this parameter
        """
        ...

    @abstractmethod
    def transform_weight(self, key: str, array: np.ndarray) -> np.ndarray:
        """Transform weight array from PyTorch to MLX format.

        Common transformations:
        - Transpose conv kernels: PyTorch [O, I, K] -> MLX [K, I, O]
        - Transpose linear layers: PyTorch [O, I] -> MLX [I, O]

        Args:
            key: Parameter key (can be used to decide transformation)
            array: NumPy array of weights

        Returns:
            Transformed NumPy array
        """
        ...

    def convert(
        self,
        pytorch_path: str | Path,
        output_path: str | Path,
        config: dict[str, Any] | None = None,
    ) -> dict[str, mx.array]:
        """Convert PyTorch weights to MLX safetensors format.

        Args:
            pytorch_path: Path to PyTorch checkpoint (.pt, .pth, .th, .ckpt)
            output_path: Output path for .safetensors file
            config: Optional config dict to save alongside weights

        Returns:
            Dictionary of converted MLX arrays

        Raises:
            WeightConversionError: If PyTorch is not installed or conversion fails
        """
        # Import torch (may not be installed)
        try:
            import torch
        except ImportError:
            raise WeightConversionError(
                "PyTorch is required for weight conversion. "
                "Install with: pip install torch"
            )

        pytorch_path = Path(pytorch_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load checkpoint
        self._log(f"Loading PyTorch checkpoint from {pytorch_path}...")
        try:
            checkpoint = torch.load(
                pytorch_path, map_location="cpu", weights_only=False
            )
        except Exception as e:
            raise WeightConversionError(
                f"Failed to load PyTorch checkpoint: {e}"
            ) from e

        # Extract state dict
        state_dict = self._extract_state_dict(checkpoint)

        # Convert weights
        self._log(f"Converting {len(state_dict)} parameters...")
        mlx_weights = {}
        skipped = []
        converted = 0

        for pt_key, pt_tensor in state_dict.items():
            # Map key
            mlx_key = self.map_key(pt_key)
            if mlx_key is None:
                skipped.append(pt_key)
                continue

            # Convert to numpy
            np_array = pt_tensor.detach().cpu().numpy()

            # Transform weight
            np_array = self.transform_weight(pt_key, np_array)

            # Convert to MLX
            mlx_weights[mlx_key] = mx.array(np_array)
            converted += 1

        if skipped and self.verbose:
            self._log(f"Skipped {len(skipped)} parameters (not mapped)")
            if len(skipped) <= 10:
                for key in skipped:
                    self._log(f"  - {key}")
            else:
                for key in skipped[:5]:
                    self._log(f"  - {key}")
                self._log(f"  ... and {len(skipped) - 5} more")

        self._log(f"Converted {converted} parameters")

        # Save weights
        self._log(f"Saving to {output_path}...")
        mx.save_safetensors(str(output_path), mlx_weights)

        # Save config if provided
        if config is not None:
            config_path = output_path.with_name("config.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            self._log(f"Saved config to {config_path}")

        self._log("Conversion complete!")
        return mlx_weights

    def _extract_state_dict(self, checkpoint: Any) -> dict[str, Any]:
        """Extract state dict from various checkpoint formats.

        Handles common PyTorch checkpoint structures:
        - Direct state dict
        - {"state_dict": ...} (PyTorch Lightning)
        - {"model": ...}
        - {"state": ...}
        - Model object with state_dict() method
        """
        if isinstance(checkpoint, dict):
            # Try common nested formats
            for key in ("state_dict", "model", "state", "model_state_dict"):
                if key in checkpoint:
                    return checkpoint[key]
            # Assume it's already a state dict
            return checkpoint
        else:
            # Try to get state_dict from model object
            if hasattr(checkpoint, "state_dict"):
                return checkpoint.state_dict()
            raise WeightConversionError(
                f"Unknown checkpoint format: {type(checkpoint).__name__}"
            )

    def _log(self, message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(message)


class IdentityConverter(WeightConverter):
    """Simple converter that passes through keys and weights unchanged.

    Useful for models where PyTorch and MLX use the same format.
    """

    model_name = "identity"

    def __init__(
        self,
        skip_patterns: list[str] | None = None,
        key_transform: Callable[[str], str] | None = None,
        verbose: bool = True,
    ):
        """Initialize identity converter.

        Args:
            skip_patterns: List of substrings; keys containing these are skipped
            key_transform: Optional function to transform key names
            verbose: Print progress messages
        """
        super().__init__(verbose=verbose)
        self.skip_patterns = skip_patterns or []
        self.key_transform = key_transform

    def map_key(self, pt_key: str) -> str | None:
        for pattern in self.skip_patterns:
            if pattern in pt_key:
                return None
        if self.key_transform:
            return self.key_transform(pt_key)
        return pt_key

    def transform_weight(self, key: str, array: np.ndarray) -> np.ndarray:
        return array


__all__ = ["WeightConverter", "IdentityConverter", "WeightConversionError"]
