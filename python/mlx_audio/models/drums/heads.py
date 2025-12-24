"""Prediction heads for onset detection and velocity estimation.

The model has two output heads:
1. Onset head: Binary classification for each frame/class (is there a drum hit?)
2. Velocity head: Regression for velocity (0-127) when there is a hit

Both heads operate per-frame, allowing frame-level drum transcription.

Ported from PyTorch implementation.
"""

import mlx.core as mx
import mlx.nn as nn


class OnsetHead(nn.Module):
    """Onset detection head.

    Predicts probability of drum onset for each class at each frame.
    Output is passed through sigmoid for binary classification.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int = 14,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        """Initialize onset head.

        Args:
            embed_dim: Input embedding dimension
            num_classes: Number of drum classes
            hidden_dim: Hidden layer dimension (defaults to embed_dim // 2)
            dropout: Dropout rate
        """
        super().__init__()

        hidden_dim = hidden_dim or embed_dim // 2

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input features (batch, seq_len, embed_dim)

        Returns:
            Onset logits (batch, seq_len, num_classes)
            Note: Returns logits, apply sigmoid for probabilities
        """
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class VelocityHead(nn.Module):
    """Velocity prediction head.

    Predicts velocity (0-127) for each class at each frame.
    Velocity is only meaningful where there is an onset.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int = 14,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        """Initialize velocity head.

        Args:
            embed_dim: Input embedding dimension
            num_classes: Number of drum classes
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()

        hidden_dim = hidden_dim or embed_dim // 2

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input features (batch, seq_len, embed_dim)

        Returns:
            Velocity predictions (batch, seq_len, num_classes) in [0, 1]
            Multiply by 127 to get MIDI velocity
        """
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = mx.sigmoid(x)
        return x


class DualHead(nn.Module):
    """Combined onset and velocity prediction head.

    Optionally shares some layers between onset and velocity prediction.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int = 14,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        share_layers: bool = False,
    ):
        """Initialize dual head.

        Args:
            embed_dim: Input embedding dimension
            num_classes: Number of drum classes
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            share_layers: Whether to share initial layers
        """
        super().__init__()

        hidden_dim = hidden_dim or embed_dim // 2
        self.share_layers = share_layers

        if share_layers:
            # Shared feature extraction
            self.shared_fc = nn.Linear(embed_dim, hidden_dim)
            self.shared_dropout = nn.Dropout(dropout)

            # Separate output layers
            self.onset_out = nn.Linear(hidden_dim, num_classes)
            self.velocity_fc = nn.Linear(hidden_dim, num_classes)
        else:
            # Separate heads
            self.onset_head = OnsetHead(embed_dim, num_classes, hidden_dim, dropout)
            self.velocity_head = VelocityHead(embed_dim, num_classes, hidden_dim, dropout)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Forward pass.

        Args:
            x: Input features (batch, seq_len, embed_dim)

        Returns:
            Tuple of:
                - onset_logits (batch, seq_len, num_classes)
                - velocity (batch, seq_len, num_classes) in [0, 1]
        """
        if self.share_layers:
            shared = self.shared_fc(x)
            shared = nn.gelu(shared)
            shared = self.shared_dropout(shared)

            onset_logits = self.onset_out(shared)
            velocity = mx.sigmoid(self.velocity_fc(shared))
        else:
            onset_logits = self.onset_head(x)
            velocity = self.velocity_head(x)

        return onset_logits, velocity
