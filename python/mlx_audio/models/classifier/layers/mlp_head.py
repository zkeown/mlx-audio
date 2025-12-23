"""MLP classifier head for audio classification."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.models.classifier.config import MLPHeadConfig


class MLPHead(nn.Module):
    """MLP classifier head for audio classification/tagging.

    A multi-layer perceptron that maps CLAP embeddings to class logits.
    Supports configurable hidden layers, dropout, and batch normalization.

    Architecture:
        Input (512) -> [Linear -> BN? -> Act -> Dropout] * N -> Linear -> Output

    Args:
        config: MLPHeadConfig specifying the architecture

    Example:
        >>> config = MLPHeadConfig(input_dim=512, num_classes=50, hidden_dims=[256])
        >>> head = MLPHead(config)
        >>> embeddings = mx.random.normal((4, 512))
        >>> logits = head(embeddings)  # [4, 50]
    """

    def __init__(self, config: MLPHeadConfig) -> None:
        super().__init__()
        self.config = config

        # Build layers
        self._layers: list[nn.Module] = []
        prev_dim = config.input_dim

        # Hidden layers
        for i, hidden_dim in enumerate(config.hidden_dims):
            # Linear layer
            self._layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization (optional)
            if config.use_batch_norm:
                self._layers.append(nn.BatchNorm(hidden_dim))

            # Activation
            self._layers.append(self._get_activation(config.activation))

            # Dropout
            if config.dropout > 0:
                self._layers.append(nn.Dropout(config.dropout))

            prev_dim = hidden_dim

        # Output layer (no activation, produces logits)
        self._output = nn.Linear(prev_dim, config.num_classes)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation module by name.

        Args:
            name: Activation name ("relu", "gelu", "silu")

        Returns:
            Activation module
        """
        if name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU()
        elif name == "silu":
            return nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {name}")

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through the MLP head.

        Args:
            x: Input embeddings of shape [B, input_dim]

        Returns:
            Logits of shape [B, num_classes]
        """
        for layer in self._layers:
            x = layer(x)
        return self._output(x)

    @property
    def num_classes(self) -> int:
        """Number of output classes."""
        return self.config.num_classes

    @property
    def input_dim(self) -> int:
        """Expected input dimension."""
        return self.config.input_dim
