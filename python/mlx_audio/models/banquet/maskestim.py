"""Mask Estimation module for Banquet model."""

from __future__ import annotations

from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.models.banquet.utils import (
    MusicalBandsplitSpecification,
    band_widths_from_specs,
    check_no_gap,
    check_nonzero_bandwidth,
)


class GLU(nn.Module):
    """Gated Linear Unit activation."""

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def __call__(self, x: mx.array) -> mx.array:
        a, b = mx.split(x, 2, axis=self.dim)
        return a * mx.sigmoid(b)


class NormMLP(nn.Module):
    """Normalization + MLP for per-band mask estimation.

    Architecture:
        Input -> LayerNorm -> Linear -> Tanh -> Linear -> GLU -> Reshape
    """

    def __init__(
        self,
        emb_dim: int,
        mlp_dim: int,
        bandwidth: int,
        in_channel: int,
        hidden_activation: str = "Tanh",
        complex_mask: bool = True,
    ) -> None:
        super().__init__()

        self.bandwidth = bandwidth
        self.in_channel = in_channel
        self.complex_mask = complex_mask
        self.reim = 2 if complex_mask else 1

        self.norm = nn.LayerNorm(emb_dim)

        # Hidden layer
        self.hidden_linear = nn.Linear(emb_dim, mlp_dim)
        if hidden_activation == "Tanh":
            self.hidden_activation = nn.Tanh()
        elif hidden_activation == "ReLU":
            self.hidden_activation = nn.ReLU()
        elif hidden_activation == "GELU":
            self.hidden_activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {hidden_activation}")

        # Output layer with GLU (doubles output dim, then halves with GLU)
        output_dim = bandwidth * in_channel * self.reim * 2
        self.output_linear = nn.Linear(mlp_dim, output_dim)
        self.glu = GLU(dim=-1)

    def __call__(self, qb: mx.array) -> mx.array:
        """Forward pass.

        Args:
            qb: Band embeddings [batch, n_time, emb_dim]

        Returns:
            Band mask [batch, in_channel, bandwidth, n_time]
        """
        batch, n_time, _ = qb.shape

        # Apply norm and hidden layer
        qb = self.norm(qb)
        qb = self.hidden_linear(qb)
        qb = self.hidden_activation(qb)

        # Output with GLU
        mb = self.output_linear(qb)
        mb = self.glu(mb)  # [batch, n_time, bandwidth * in_channel * reim]

        # Reshape to [batch, n_time, in_channel, bandwidth, reim]
        if self.complex_mask:
            mb = mb.reshape(batch, n_time, self.in_channel, self.bandwidth, self.reim)
            # Combine real/imag into complex: [batch, n_time, in_channel, bandwidth]
            # For MLX, we keep as real with last dim 2 for now
            # Permute to [batch, in_channel, bandwidth, n_time, reim]
            mb = mx.transpose(mb, (0, 2, 3, 1, 4))
        else:
            mb = mb.reshape(batch, n_time, self.in_channel, self.bandwidth)
            mb = mx.transpose(mb, (0, 2, 3, 1))

        return mb


class OverlappingMaskEstimationModule(nn.Module):
    """Overlapping mask estimation with frequency weighting.

    Processes each band through a NormMLP and accumulates masks
    with frequency weights for overlapping bands.
    """

    def __init__(
        self,
        in_channel: int,
        band_specs: List[Tuple[int, int]],
        freq_weights: List[mx.array],
        n_freq: int,
        emb_dim: int,
        mlp_dim: int,
        hidden_activation: str = "Tanh",
        complex_mask: bool = True,
        use_freq_weights: bool = True,
    ) -> None:
        super().__init__()

        check_nonzero_bandwidth(band_specs)
        check_no_gap(band_specs)

        self.band_specs = band_specs
        self.band_widths = band_widths_from_specs(band_specs)
        self.n_bands = len(band_specs)
        self.n_freq = n_freq
        self.in_channel = in_channel
        self.complex_mask = complex_mask
        self.use_freq_weights = use_freq_weights

        # Store frequency weights
        self.freq_weights = freq_weights

        # Per-band NormMLP modules
        self.norm_mlp = [
            NormMLP(
                emb_dim=emb_dim,
                mlp_dim=mlp_dim,
                bandwidth=bw,
                in_channel=in_channel,
                hidden_activation=hidden_activation,
                complex_mask=complex_mask,
            )
            for bw in self.band_widths
        ]

    def __call__(self, q: mx.array) -> mx.array:
        """Forward pass.

        Args:
            q: Band embeddings [batch, n_bands, n_time, emb_dim]

        Returns:
            Full-frequency mask [batch, in_channel, n_freq, n_time] or
            [batch, in_channel, n_freq, n_time, 2] for complex mask
        """
        batch, n_bands, n_time, emb_dim = q.shape

        # Compute per-band masks
        mask_list = []
        for b, nmlp in enumerate(self.norm_mlp):
            qb = q[:, b, :, :]  # [batch, n_time, emb_dim]
            mb = nmlp(qb)  # [batch, in_channel, bandwidth, n_time] or with reim
            mask_list.append(mb)

        # Accumulate masks into full frequency range
        if self.complex_mask:
            masks = mx.zeros((batch, self.in_channel, self.n_freq, n_time, 2))
        else:
            masks = mx.zeros((batch, self.in_channel, self.n_freq, n_time))

        for im, mask in enumerate(mask_list):
            fstart, fend = self.band_specs[im]

            if self.use_freq_weights and self.freq_weights is not None:
                fw = self.freq_weights[im]
                if self.complex_mask:
                    # fw: [bandwidth] -> [1, 1, bandwidth, 1, 1]
                    fw = fw.reshape(1, 1, -1, 1, 1)
                else:
                    fw = fw.reshape(1, 1, -1, 1)
                mask = mask * fw

            # Accumulate into full mask
            if self.complex_mask:
                # masks[:, :, fstart:fend, :, :] += mask
                # MLX doesn't support slice assignment, so we use scatter
                indices = mx.arange(fstart, fend)
                masks = masks.at[:, :, fstart:fend, :, :].add(mask)
            else:
                masks = masks.at[:, :, fstart:fend, :].add(mask)

        return masks

    @classmethod
    def from_config(
        cls,
        n_fft: int = 2048,
        sample_rate: int = 44100,
        n_bands: int = 64,
        in_channel: int = 2,
        emb_dim: int = 128,
        mlp_dim: int = 512,
        hidden_activation: str = "Tanh",
        complex_mask: bool = True,
        use_freq_weights: bool = True,
    ) -> "OverlappingMaskEstimationModule":
        """Create OverlappingMaskEstimationModule from configuration."""
        spec = MusicalBandsplitSpecification(
            nfft=n_fft,
            fs=sample_rate,
            n_bands=n_bands,
        )

        return cls(
            in_channel=in_channel,
            band_specs=spec.get_band_specs(),
            freq_weights=spec.get_freq_weights(),
            n_freq=n_fft // 2 + 1,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            hidden_activation=hidden_activation,
            complex_mask=complex_mask,
            use_freq_weights=use_freq_weights,
        )
