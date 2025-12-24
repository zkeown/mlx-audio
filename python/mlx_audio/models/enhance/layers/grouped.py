"""Grouped layers for efficient computation."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class StackedGRU(nn.Module):
    """Multi-layer GRU using stacked single-layer GRUs (MLX compatibility).

    MLX's nn.GRU doesn't support num_layers, so we stack individual GRUs.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.layers = [
            nn.GRU(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ]

    def __call__(
        self,
        x: mx.array,
        hidden: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Forward pass through stacked GRU layers."""
        batch = x.shape[0]
        new_hiddens = []
        for i, gru in enumerate(self.layers):
            h = hidden[i] if hidden is not None else None
            result = gru(x, h)
            # MLX GRU returns (output, hidden) if hidden provided, else output
            if isinstance(result, tuple):
                x, h_new = result
            else:
                x = result
                h_new = mx.zeros((batch, self.hidden_size))
            new_hiddens.append(h_new)
        return x, mx.stack(new_hiddens, axis=0)


class GroupedLinear(nn.Module):
    """Grouped linear layer.

    Splits input into groups and applies smaller linear transforms to each,
    reducing computation from O(H^2) to O(H^2/G).

    Parameters
    ----------
    input_size : int
        Input dimension.
    output_size : int
        Output dimension.
    num_groups : int, default=1
        Number of groups. Must divide both input_size and output_size.
    bias : bool, default=True
        Whether to include bias.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.input_size = input_size
        self.output_size = output_size

        assert input_size % num_groups == 0
        assert output_size % num_groups == 0

        group_in = input_size // num_groups
        group_out = output_size // num_groups

        # Create weight for all groups: (num_groups, group_out, group_in)
        self.weight = mx.random.normal(
            shape=(num_groups, group_out, group_in)
        ) * (2.0 / (group_in + group_out)) ** 0.5

        if bias:
            self.bias = mx.zeros((output_size,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Parameters
        ----------
        x : mx.array
            Input tensor of shape (..., input_size).

        Returns
        -------
        mx.array
            Output tensor of shape (..., output_size).
        """
        # Get original shape
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.input_size)
        batch = x_flat.shape[0]

        group_in = self.input_size // self.num_groups

        # Reshape to groups: (batch, num_groups, group_in)
        x_grouped = x_flat.reshape(batch, self.num_groups, group_in)

        # Apply per-group linear using batched matmul (vectorized, no loop)
        # x_grouped: (batch, num_groups, group_in)
        # weight: (num_groups, group_out, group_in)
        # Result: (batch, num_groups, group_out)
        out_grouped = mx.einsum(
            "bgi,gio->bgo", x_grouped, self.weight.swapaxes(-1, -2)
        )

        # Reshape back: (batch, output_size)
        out = out_grouped.reshape(batch, self.output_size)

        if self.bias is not None:
            out = out + self.bias

        # Restore original batch dimensions
        return out.reshape(*orig_shape, self.output_size)


class GroupedGRU(nn.Module):
    """Grouped GRU for efficient sequence modeling.

    Splits input/hidden into groups, applies smaller GRUs to each,
    then shuffles outputs for inter-group information flow.

    Parameters
    ----------
    input_size : int
        Input feature dimension.
    hidden_size : int
        Hidden state dimension.
    num_groups : int, default=1
        Number of groups.
    num_layers : int, default=1
        Number of stacked GRU layers.
    batch_first : bool, default=True
        If True, input shape is (batch, seq, features).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_groups: int = 1,
        num_layers: int = 1,
        batch_first: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.num_layers = num_layers
        self.batch_first = batch_first

        # For simplicity, use standard GRU when num_groups=1
        if num_groups == 1:
            if num_layers == 1:
                self.gru = nn.GRU(input_size, hidden_size)
            else:
                self.gru = StackedGRU(input_size, hidden_size, num_layers)
        else:
            # Create per-group GRUs
            assert hidden_size % num_groups == 0
            group_hidden = hidden_size // num_groups

            # Input projection to group size
            self.input_proj = GroupedLinear(
                input_size, hidden_size, num_groups
            )

            # Per-group GRUs (single layer each for simplicity)
            self.grus = [
                nn.GRU(group_hidden, group_hidden)
                for _ in range(num_groups)
            ]

            # Shuffle layer for inter-group mixing
            self.shuffle = nn.Linear(hidden_size, hidden_size)

    def __call__(
        self,
        x: mx.array,
        hidden: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Forward pass.

        Parameters
        ----------
        x : mx.array
            Input tensor of shape (batch, seq, input_size) if batch_first.
        hidden : mx.array, optional
            Initial hidden state.

        Returns
        -------
        tuple
            (output, hidden) where output is (batch, seq, hidden_size)
            and hidden is the final hidden state.
        """
        if self.num_groups == 1:
            result = self.gru(x, hidden)
            # Handle MLX GRU return (output only if no hidden, else tuple)
            if isinstance(result, tuple):
                return result
            else:
                batch = x.shape[0]
                return result, mx.zeros((batch, self.hidden_size))

        # Grouped processing
        if self.batch_first:
            batch, seq, _ = x.shape
        else:
            seq, batch, _ = x.shape
            x = mx.transpose(x, (1, 0, 2))

        group_hidden = self.hidden_size // self.num_groups

        # Project input to hidden size
        x_proj = self.input_proj(x)  # (batch, seq, hidden_size)

        # Split into groups
        x_groups = x_proj.reshape(batch, seq, self.num_groups, group_hidden)

        # Process each group
        outputs = []
        new_hiddens = []

        for g in range(self.num_groups):
            x_g = x_groups[:, :, g, :]  # (batch, seq, group_hidden)

            if hidden is not None:
                h_g = hidden[:, g * group_hidden:(g + 1) * group_hidden]
            else:
                h_g = None

            result = self.grus[g](x_g, h_g)
            # Handle MLX GRU return
            if isinstance(result, tuple):
                out_g, h_g_new = result
            else:
                out_g = result
                h_g_new = mx.zeros((batch, group_hidden))
            outputs.append(out_g)
            new_hiddens.append(h_g_new)

        # Concatenate groups
        output = mx.concatenate(outputs, axis=-1)

        # Channel shuffle for inter-group mixing
        output = self.shuffle(output)

        # Combine hidden states
        new_hidden = mx.concatenate(new_hiddens, axis=-1)

        if not self.batch_first:
            output = mx.transpose(output, (1, 0, 2))

        return output, new_hidden
