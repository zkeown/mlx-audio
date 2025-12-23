"""Sequential Band Modelling (SeqBand) module for Banquet model."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class ResidualRNN(nn.Module):
    """Residual RNN block with LayerNorm and bidirectional processing.

    Architecture: Input -> LayerNorm -> BiLSTM -> FC -> + Input (residual)
    """

    def __init__(
        self,
        emb_dim: int,
        rnn_dim: int,
        bidirectional: bool = True,
        rnn_type: str = "LSTM",
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        self.use_layer_norm = use_layer_norm
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type

        if use_layer_norm:
            self.norm = nn.LayerNorm(emb_dim)
        else:
            self.norm = nn.GroupNorm(num_groups=emb_dim, dims=emb_dim)

        # Create RNN layers
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(emb_dim, rnn_dim)
            if bidirectional:
                self.rnn_reverse = nn.LSTM(emb_dim, rnn_dim)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(emb_dim, rnn_dim)
            if bidirectional:
                self.rnn_reverse = nn.GRU(emb_dim, rnn_dim)
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")

        # Output projection
        fc_in = rnn_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_in, emb_dim)

    def __call__(self, z: mx.array) -> mx.array:
        """Forward pass.

        Args:
            z: Input [batch, n_uncrossed, n_across, emb_dim]

        Returns:
            Output with same shape as input
        """
        z0 = z  # Save for residual

        # Apply normalization
        if self.use_layer_norm:
            z = self.norm(z)
        else:
            # GroupNorm expects [batch, channels, ...]
            z = mx.transpose(z, (0, 3, 1, 2))
            z = self.norm(z)
            z = mx.transpose(z, (0, 2, 3, 1))

        batch, n_uncrossed, n_across, emb_dim = z.shape

        # Use batch trick: flatten first two dims for RNN processing
        z = z.reshape(batch * n_uncrossed, n_across, emb_dim)

        # Apply RNN
        if self.rnn_type == "LSTM":
            rnn_out, _ = self.rnn(z)
        else:  # GRU
            rnn_out, _ = self.rnn(z)

        if self.bidirectional:
            # Process reversed sequence
            z_rev = z[:, ::-1, :]
            if self.rnn_type == "LSTM":
                rnn_rev_out, _ = self.rnn_reverse(z_rev)
            else:
                rnn_rev_out, _ = self.rnn_reverse(z_rev)
            # Reverse back and concatenate
            rnn_rev_out = rnn_rev_out[:, ::-1, :]
            z = mx.concatenate([rnn_out, rnn_rev_out], axis=-1)
        else:
            z = rnn_out

        # Reshape back
        z = z.reshape(batch, n_uncrossed, n_across, -1)

        # Project to embedding dimension
        z = self.fc(z)

        # Residual connection
        z = z + z0

        return z


class SeqBandModellingModule(nn.Module):
    """Sequential Band Modelling module.

    Alternates between time and frequency RNNs to model temporal and
    cross-band dependencies.

    Architecture:
    - n_modules pairs of RNNs
    - Each pair: Time RNN -> transpose -> Freq RNN -> transpose
    - Total: 2 * n_modules ResidualRNN layers
    """

    def __init__(
        self,
        n_modules: int = 12,
        emb_dim: int = 128,
        rnn_dim: int = 256,
        bidirectional: bool = True,
        rnn_type: str = "LSTM",
    ) -> None:
        super().__init__()

        self.n_modules = n_modules

        # Create 2 * n_modules RNN layers (alternating time and freq)
        self.seqband = [
            ResidualRNN(
                emb_dim=emb_dim,
                rnn_dim=rnn_dim,
                bidirectional=bidirectional,
                rnn_type=rnn_type,
            )
            for _ in range(2 * n_modules)
        ]

    def __call__(self, z: mx.array) -> mx.array:
        """Forward pass.

        Args:
            z: Band embeddings [batch, n_bands, n_time, emb_dim]

        Returns:
            Processed embeddings [batch, n_bands, n_time, emb_dim]
        """
        # Process through alternating time and freq RNNs
        for sbm in self.seqband:
            z = sbm(z)
            # Transpose between time and freq dimensions
            # [batch, n_bands, n_time, emb_dim] <-> [batch, n_time, n_bands, emb_dim]
            z = mx.transpose(z, (0, 2, 1, 3))

        return z  # [batch, n_bands, n_time, emb_dim]
