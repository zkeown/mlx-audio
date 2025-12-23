"""VAD model layers.

LSTM-based layers for Voice Activity Detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    pass


class VADEncoder(nn.Module):
    """Audio encoder for VAD.

    Converts raw audio samples into a single feature vector per window.
    Uses 1D convolutions with global pooling to collapse the temporal dimension,
    matching the Silero VAD architecture where LSTM processes 1 timestep per window.

    Args:
        input_size: Number of input samples per window
        hidden_size: Output feature dimension
    """

    def __init__(self, input_size: int = 512, hidden_size: int = 128) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Convolutional feature extractor
        # Progressive downsampling to reduce sequence length
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=hidden_size // 4,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=hidden_size // 4,
            out_channels=hidden_size // 2,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv3 = nn.Conv1d(
            in_channels=hidden_size // 2,
            out_channels=hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # Layer normalization for stability
        self.norm = nn.LayerNorm(hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        """Encode audio samples to a single feature vector.

        Args:
            x: Audio samples [batch, samples] or [samples]

        Returns:
            Features [batch, hidden_size] - single vector per window
        """
        # Add batch dimension if needed
        if x.ndim == 1:
            x = x[None, :]
            squeeze_batch = True
        else:
            squeeze_batch = False

        # Add channel dimension in NLC format: [batch, samples] -> [batch, samples, 1]
        # MLX Conv1d expects NLC (batch, length, channels)
        x = x[:, :, None]

        # Apply convolutions with progressive downsampling
        # Each conv with stride=2 halves the sequence length
        x = nn.relu(self.conv1(x))  # [batch, L/2, hidden/4]
        x = nn.relu(self.conv2(x))  # [batch, L/4, hidden/2]
        x = nn.relu(self.conv3(x))  # [batch, L/8, hidden]

        # Global average pooling to collapse to single vector
        # This matches Silero architecture: 1 feature vector per audio window
        x = mx.mean(x, axis=1)  # [batch, hidden]

        # Apply normalization
        x = self.norm(x)

        if squeeze_batch:
            x = x[0]

        return x


class VADDecoder(nn.Module):
    """Decoder that produces speech probability from LSTM features.

    Args:
        hidden_size: Input feature dimension from LSTM
    """

    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        # Classification head
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def __call__(self, x: mx.array) -> mx.array:
        """Decode LSTM output to speech probability.

        Args:
            x: LSTM output [batch, hidden_size] or [hidden_size]

        Returns:
            Speech probability [batch, 1] or scalar
        """
        x = nn.relu(self.fc1(x))
        x = mx.sigmoid(self.fc2(x))
        return x


class StackedLSTM(nn.Module):
    """Stacked LSTM layers with proper hidden state handling.

    MLX's LSTM doesn't directly support num_layers, so we stack manually.

    Args:
        input_size: Feature dimension of input
        hidden_size: LSTM hidden state dimension
        num_layers: Number of stacked LSTM layers
        bias: Whether to use bias terms
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create stacked LSTM layers
        self.layers = []
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.layers.append(
                nn.LSTM(layer_input_size, hidden_size, bias=bias)
            )

    def __call__(
        self,
        x: mx.array,
        hidden: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        """Forward pass through stacked LSTM.

        Args:
            x: Input tensor [batch, seq_len, input_size] or [seq_len, input_size]
            hidden: Optional tuple of (h, c) each with shape
                    [num_layers, batch, hidden_size] or [num_layers, hidden_size]

        Returns:
            Tuple of:
                - Output tensor [batch, seq_len, hidden_size]
                - Tuple of (h, c) final hidden states
        """
        # Handle batch dimension
        if x.ndim == 2:
            x = x[None, :]
            squeeze_batch = True
        else:
            squeeze_batch = False

        batch_size = x.shape[0]

        # Initialize hidden states if not provided
        if hidden is None:
            h_list = [
                mx.zeros((batch_size, self.hidden_size))
                for _ in range(self.num_layers)
            ]
            c_list = [
                mx.zeros((batch_size, self.hidden_size))
                for _ in range(self.num_layers)
            ]
        else:
            # Unpack stacked hidden states
            h_stacked, c_stacked = hidden
            h_list = [h_stacked[i] for i in range(self.num_layers)]
            c_list = [c_stacked[i] for i in range(self.num_layers)]

        # Process through each layer
        new_h_list = []
        new_c_list = []

        for i, lstm in enumerate(self.layers):
            # MLX LSTM takes separate hidden and cell arguments
            h_seq, c_seq = lstm(x, hidden=h_list[i], cell=c_list[i])

            # The output is the sequence of hidden states
            x = h_seq

            # Keep final hidden states for next call
            new_h_list.append(h_seq[:, -1, :])  # Last time step
            new_c_list.append(c_seq[:, -1, :])

        # Stack hidden states back: [num_layers, batch, hidden]
        h_out = mx.stack(new_h_list, axis=0)
        c_out = mx.stack(new_c_list, axis=0)

        if squeeze_batch:
            x = x[0]
            h_out = h_out[:, 0, :]
            c_out = c_out[:, 0, :]

        return x, (h_out, c_out)


__all__ = ["VADEncoder", "VADDecoder", "StackedLSTM"]
