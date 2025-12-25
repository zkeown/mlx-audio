"""BandSplit module for Banquet model."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.models.banquet.utils import (
    MusicalBandsplitSpecification,
    band_widths_from_specs,
    check_no_gap,
    check_no_overlap,
    check_nonzero_bandwidth,
)


class NormFC(nn.Module):
    """Normalization + Fully Connected layer for band processing.

    Applies LayerNorm over flattened band input, then projects to embedding dimension.
    """

    def __init__(
        self,
        emb_dim: int,
        bandwidth: int,
        in_channel: int,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
    ) -> None:
        super().__init__()

        if normalize_channel_independently:
            raise NotImplementedError("normalize_channel_independently not supported")

        self.treat_channel_as_feature = treat_channel_as_feature
        reim = 2  # Real and imaginary parts

        # LayerNorm over flattened input
        self.norm = nn.LayerNorm(in_channel * bandwidth * reim)

        # FC input dimension depends on treat_channel_as_feature
        fc_in = bandwidth * reim
        if treat_channel_as_feature:
            fc_in *= in_channel
        else:
            assert emb_dim % in_channel == 0
            emb_dim = emb_dim // in_channel

        self.fc = nn.Linear(fc_in, emb_dim)

    def __call__(self, xb: mx.array) -> mx.array:
        """Forward pass.

        Args:
            xb: Band input [batch, n_time, in_chan, reim * bandwidth]

        Returns:
            Band embeddings [batch, n_time, emb_dim]
        """
        batch, n_time, in_chan, ribw = xb.shape

        # Flatten for LayerNorm: [batch, n_time, in_chan * reim * bandwidth]
        xb = self.norm(xb.reshape(batch, n_time, in_chan * ribw))

        if not self.treat_channel_as_feature:
            # Reshape back: [batch, n_time, in_chan, reim * bandwidth]
            xb = xb.reshape(batch, n_time, in_chan, ribw)

        # Project to embedding dimension
        zb = self.fc(xb)

        if not self.treat_channel_as_feature:
            # Flatten channel dimension into embedding
            batch, n_time, in_chan, emb_dim_per_chan = zb.shape
            zb = zb.reshape(batch, n_time, in_chan * emb_dim_per_chan)

        return zb  # [batch, n_time, emb_dim]


class BandSplitModule(nn.Module):
    """Band splitting module for frequency decomposition.

    Splits complex spectrogram into frequency bands and projects each band
    to an embedding dimension using per-band NormFC modules.
    """

    def __init__(
        self,
        in_channel: int,
        band_specs: list[tuple[int, int]],
        emb_dim: int = 128,
        require_no_overlap: bool = False,
        require_no_gap: bool = True,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
    ) -> None:
        super().__init__()

        check_nonzero_bandwidth(band_specs)

        if require_no_gap:
            check_no_gap(band_specs)

        if require_no_overlap:
            check_no_overlap(band_specs)

        self.band_specs = band_specs
        self.band_widths = band_widths_from_specs(band_specs)
        self.n_bands = len(band_specs)
        self.emb_dim = emb_dim

        # Create per-band NormFC modules
        self.norm_fc_modules = [
            NormFC(
                emb_dim=emb_dim,
                bandwidth=bw,
                in_channel=in_channel,
                normalize_channel_independently=normalize_channel_independently,
                treat_channel_as_feature=treat_channel_as_feature,
            )
            for bw in self.band_widths
        ]

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Complex spectrogram [batch, in_chan, n_freq, n_time]
               Note: x should be complex-valued

        Returns:
            Band embeddings [batch, n_bands, n_time, emb_dim]
        """
        batch, in_chan, n_freq, n_time = x.shape

        # Convert complex to real/imag: [batch, in_chan, n_freq, n_time, 2]
        # If x is already real with last dim 2, use as-is
        if x.dtype in (mx.complex64,):
            xr = mx.stack([x.real, x.imag], axis=-1)
        else:
            # Assume x is [batch, in_chan, n_freq, n_time] with interleaved or
            # already has trailing dim
            xr = mx.stack([x.real, x.imag], axis=-1)

        # Permute: [batch, n_time, in_chan, 2, n_freq]
        xr = mx.transpose(xr, (0, 3, 1, 4, 2))

        batch, n_time, in_chan, reim, _ = xr.shape

        # Initialize output
        z = mx.zeros((batch, self.n_bands, n_time, self.emb_dim))

        # Process each band
        outputs = []
        for i, nfm in enumerate(self.norm_fc_modules):
            fstart, fend = self.band_specs[i]
            # Extract band: [batch, n_time, in_chan, reim, bandwidth]
            xb = xr[..., fstart:fend]
            # Flatten reim and bandwidth: [batch, n_time, in_chan, reim * bandwidth]
            xb = xb.reshape(batch, n_time, in_chan, -1)
            # Apply NormFC
            zb = nfm(xb)  # [batch, n_time, emb_dim]
            outputs.append(zb)

        # Stack bands: [batch, n_bands, n_time, emb_dim]
        z = mx.stack(outputs, axis=1)

        return z

    @classmethod
    def from_config(
        cls,
        n_fft: int = 2048,
        sample_rate: int = 44100,
        n_bands: int = 64,
        in_channel: int = 2,
        emb_dim: int = 128,
        band_type: str = "musical",
        **kwargs,
    ) -> BandSplitModule:
        """Create BandSplitModule from configuration.

        Args:
            n_fft: FFT size
            sample_rate: Sample rate in Hz
            n_bands: Number of frequency bands
            in_channel: Number of audio channels
            emb_dim: Embedding dimension
            band_type: Band specification type ("musical")
            **kwargs: Additional arguments passed to __init__

        Returns:
            Configured BandSplitModule
        """
        if band_type != "musical":
            raise ValueError(f"Unsupported band_type: {band_type}")

        spec = MusicalBandsplitSpecification(
            nfft=n_fft,
            fs=sample_rate,
            n_bands=n_bands,
        )

        return cls(
            in_channel=in_channel,
            band_specs=spec.get_band_specs(),
            emb_dim=emb_dim,
            **kwargs,
        )
