"""BagOfModels ensemble for HTDemucs.

Implements the ensemble approach used by htdemucs_ft, where 4 specialized
models are combined to achieve better separation quality for each stem.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from mlx_audio.models.demucs.model import HTDemucs


class BagOfModels:
    """Ensemble of HTDemucs models for improved source separation.

    htdemucs_ft uses 4 specialized models, each trained to excel at one
    source type. The weight matrix selects which model's output to use
    for each stem:
        - Model 0 → drums
        - Model 1 → bass
        - Model 2 → other
        - Model 3 → vocals

    This provides ~3dB SI-SDR improvement over a single general-purpose model.

    Args:
        models: List of HTDemucs models (typically 4)
        weights: Weight matrix [num_models, num_sources] for combining outputs.
                 Default is identity matrix (each model → one stem).

    Example:
        >>> bag = BagOfModels.from_pretrained("mlx-community/htdemucs-ft-bag")
        >>> stems = bag(mixture)  # [B, 4, C, T]
    """

    def __init__(
        self,
        models: list[HTDemucs],
        weights: mx.array | None = None,
    ) -> None:
        self.models = models
        self.num_models = len(models)

        # Get config from first model (all models should have same config)
        self._config = models[0].config

        # Default to identity matrix: model_i contributes only to stem_i
        if weights is None:
            weights = mx.eye(self.num_models)
        self.weights = weights

    @property
    def config(self):
        """Model configuration (from first model)."""
        return self._config

    def eval(self) -> "BagOfModels":
        """Set all models to evaluation mode.

        Returns:
            self for method chaining
        """
        for model in self.models:
            model.eval()
        return self

    def __call__(self, mix: mx.array) -> mx.array:
        """Run all models and combine outputs.

        Args:
            mix: Input mixture [B, C, T] or [C, T]

        Returns:
            Separated stems [B, S, C, T] or [S, C, T]
        """
        # Handle unbatched input
        if mix.ndim == 2:
            mix = mix[None, ...]
            squeeze_batch = True
        else:
            squeeze_batch = False

        # Run all models
        outputs = []
        for model in self.models:
            out = model(mix)  # [B, S, C, T]
            outputs.append(out)

        # Stack outputs: [num_models, B, S, C, T]
        stacked = mx.stack(outputs, axis=0)

        # Apply weight matrix to combine model outputs
        # weights[m, s] indicates how much model m contributes to stem s
        # We want: result[b, s, c, t] = sum_m weights[m, s] * stacked[m, b, s, c, t]
        #
        # The weight matrix for htdemucs_ft is identity, so:
        # result[:, 0, :, :] = stacked[0, :, 0, :, :]  # drums from model 0
        # result[:, 1, :, :] = stacked[1, :, 1, :, :]  # bass from model 1
        # etc.
        #
        # Using einsum: 'mbsct,ms->bsct'
        result = mx.einsum('mbsct,ms->bsct', stacked, self.weights)

        # Ensure evaluation to prevent memory buildup
        mx.eval(result)

        if squeeze_batch:
            result = result[0]

        return result

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        num_models: int = 4,
    ) -> "BagOfModels":
        """Load a bag of pretrained HTDemucs models.

        Expected directory structure:
            path/
                model_0/
                    config.json
                    model.safetensors
                model_1/
                    ...
                model_2/
                    ...
                model_3/
                    ...
                weights.npy  (optional, defaults to identity)

        Args:
            path: Path to bag directory
            num_models: Expected number of models (default: 4)

        Returns:
            BagOfModels instance
        """
        from mlx_audio.models.demucs.model import HTDemucs

        path = Path(path)

        # Load individual models
        models = []
        for i in range(num_models):
            model_path = path / f"model_{i}"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model {i} not found at {model_path}. "
                    f"Expected {num_models} models in {path}"
                )
            model = HTDemucs.from_pretrained(model_path)
            models.append(model)

        # Load weights if available, otherwise use identity
        weights_path = path / "weights.npy"
        if weights_path.exists():
            import numpy as np
            weights = mx.array(np.load(weights_path))
        else:
            weights = mx.eye(num_models)

        return cls(models, weights)

    def save_pretrained(self, path: str | Path) -> None:
        """Save the bag of models to a directory.

        Args:
            path: Output directory
        """
        import numpy as np

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save each model
        for i, model in enumerate(self.models):
            model_path = path / f"model_{i}"
            model.save_pretrained(model_path)

        # Save weights
        weights_path = path / "weights.npy"
        np.save(weights_path, np.array(self.weights))
