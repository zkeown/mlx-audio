"""Banquet query-based source separation model.

Banquet uses a reference audio query to extract matching sounds from a mixture.
This is the main model that combines all components.

Pipeline:
    Query Audio → PaSST → 768-dim embedding
                              ↓
    Mixture → STFT → BandSplit → SeqBand (24 LSTMs) → FiLM Conditioning → MaskEstimation → iSTFT → Output
"""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from .bandsplit import BandSplitModule
from .config import BanquetConfig, PaSSTConfig
from .film import FiLM
from .maskestim import OverlappingMaskEstimationModule
from .passt import PaSST
from .tfmodel import SeqBandModellingModule
from .utils import MusicalBandsplitSpecification


class BanquetOutput(NamedTuple):
    """Output from Banquet separation.

    Attributes:
        audio: Separated audio [batch, channels, samples]
        spectrogram: Separated spectrogram [batch, channels, freq, time] (complex)
        mask: Estimated mask [batch, channels, freq, time] (complex)
    """

    audio: mx.array
    spectrogram: mx.array
    mask: mx.array


class Banquet(nn.Module):
    """Banquet query-based source separation model.

    Uses a reference audio query (encoded by PaSST) to extract matching sounds
    from a mixture audio signal.

    Args:
        config: Banquet model configuration
        passt_config: PaSST encoder configuration
    """

    def __init__(
        self,
        config: BanquetConfig | None = None,
        passt_config: PaSSTConfig | None = None,
    ):
        super().__init__()
        self.config = config or BanquetConfig()
        self.passt_config = passt_config or PaSSTConfig()

        # Generate band specifications
        band_spec = MusicalBandsplitSpecification(
            nfft=self.config.n_fft,
            fs=self.config.sample_rate,
            n_bands=self.config.n_bands,
        )
        self.band_specs = band_spec.get_band_specs()
        self.freq_weights = band_spec.get_freq_weights()
        self.band_widths = band_spec.get_band_widths()

        # Query encoder (PaSST)
        self.query_encoder = PaSST(self.passt_config)

        # Band split module
        self.band_split = BandSplitModule(
            band_specs=self.band_specs,
            in_channel=self.config.in_channel,
            emb_dim=self.config.emb_dim,
        )

        # Time-frequency modelling (BiLSTM)
        self.tf_model = SeqBandModellingModule(
            n_modules=self.config.n_sqm_modules,
            emb_dim=self.config.emb_dim,
            rnn_dim=self.config.rnn_dim,
            bidirectional=self.config.bidirectional,
            rnn_type=self.config.rnn_type,
        )

        # FiLM conditioning
        self.film = FiLM(
            cond_embedding_dim=self.config.cond_emb_dim,
            channels=self.config.emb_dim,
            additive=self.config.film_additive,
            multiplicative=self.config.film_multiplicative,
            depth=self.config.film_depth,
            channels_per_group=self.config.channels_per_group,
        )

        # Mask estimation
        self.mask_estim = OverlappingMaskEstimationModule(
            in_channel=self.config.in_channel,
            band_specs=self.band_specs,
            freq_weights=self.freq_weights,
            n_freq=self.config.freq_bins,
            emb_dim=self.config.emb_dim,
            mlp_dim=self.config.mlp_dim,
            hidden_activation=self.config.hidden_activation,
            complex_mask=self.config.complex_mask,
            use_freq_weights=self.config.use_freq_weights,
        )

        # STFT parameters
        self._n_fft = self.config.n_fft
        self._hop_length = self.config.hop_length
        self._win_length = self.config.effective_win_length

    def _stft(self, x: mx.array) -> mx.array:
        """Compute STFT of audio signal.

        Args:
            x: Audio signal [batch, channels, samples]

        Returns:
            Complex spectrogram [batch, channels, freq, time]
        """
        from mlx_audio.primitives import stft

        B, C, T = x.shape

        # Flatten batch and channels
        x_flat = x.reshape(B * C, T)

        # Compute STFT
        spec = stft(
            x_flat,
            n_fft=self._n_fft,
            hop_length=self._hop_length,
            win_length=self._win_length,
        )

        # Normalize
        spec = spec / mx.sqrt(mx.array(self._n_fft, dtype=mx.float32))

        # Reshape back: [B*C, freq, time] -> [B, C, freq, time]
        _, F, Tf = spec.shape
        spec = spec.reshape(B, C, F, Tf)

        return spec

    def _istft(self, spec: mx.array, length: int) -> mx.array:
        """Compute inverse STFT.

        Args:
            spec: Complex spectrogram [batch, channels, freq, time]
            length: Target output length

        Returns:
            Audio signal [batch, channels, samples]
        """
        from mlx_audio.primitives import istft

        B, C, F, Tf = spec.shape

        # Flatten batch and channels
        spec_flat = spec.reshape(B * C, F, Tf)

        # Compute iSTFT
        x = istft(
            spec_flat,
            hop_length=self._hop_length,
            win_length=self._win_length,
            n_fft=self._n_fft,
            length=length,
        )

        # Denormalize
        x = x * mx.sqrt(mx.array(self._n_fft, dtype=mx.float32))

        # Reshape back
        x = x.reshape(B, C, -1)

        # Trim to exact length
        x = x[:, :, :length]

        return x

    def _apply_complex_mask(
        self, spec: mx.array, mask: mx.array
    ) -> mx.array:
        """Apply complex mask to spectrogram.

        Args:
            spec: Complex spectrogram [batch, channels, freq, time]
            mask: Complex mask [batch, channels, freq, time, 2] (real, imag)

        Returns:
            Masked spectrogram [batch, channels, freq, time]
        """
        if self.config.complex_mask:
            # Convert mask from [B, C, F, T, 2] to complex
            mask_real = mask[..., 0]
            mask_imag = mask[..., 1]

            # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            spec_real = mx.real(spec)
            spec_imag = mx.imag(spec)

            out_real = spec_real * mask_real - spec_imag * mask_imag
            out_imag = spec_real * mask_imag + spec_imag * mask_real

            return out_real + 1j * out_imag
        else:
            # Real mask: simple multiplication
            return spec * mask

    def encode_query(self, query_mel: mx.array) -> mx.array:
        """Encode query audio using PaSST.

        Args:
            query_mel: Query mel spectrogram [batch, 1, n_mels, time]

        Returns:
            Query embedding [batch, 768]
        """
        return self.query_encoder(query_mel)

    def __call__(
        self,
        mixture: mx.array,
        query_embedding: mx.array,
    ) -> BanquetOutput:
        """Forward pass for Banquet separation.

        Args:
            mixture: Mixture audio [batch, channels, samples]
            query_embedding: Pre-computed query embedding [batch, 768]

        Returns:
            BanquetOutput with separated audio, spectrogram, and mask
        """
        # Store original length for reconstruction
        original_length = mixture.shape[-1]

        # Compute STFT of mixture
        spec = self._stft(mixture)  # [B, C, F, T]

        # Band split: spec -> band embeddings
        z = self.band_split(spec)  # [B, n_bands, n_time, emb_dim]

        # Time-frequency modelling
        z = self.tf_model(z)  # [B, n_bands, n_time, emb_dim]

        # Prepare for FiLM: [B, n_bands, n_time, emb_dim] -> [B, emb_dim, n_bands, n_time]
        z = mx.transpose(z, (0, 3, 1, 2))

        # Apply FiLM conditioning
        z = self.film(z, query_embedding)

        # Back to [B, n_bands, n_time, emb_dim]
        z = mx.transpose(z, (0, 2, 3, 1))

        # Mask estimation
        mask = self.mask_estim(z)  # [B, C, F, T, 2] or [B, C, F, T]

        # Apply mask to spectrogram
        masked_spec = self._apply_complex_mask(spec, mask)

        # Inverse STFT to get audio
        audio = self._istft(masked_spec, original_length)

        return BanquetOutput(
            audio=audio,
            spectrogram=masked_spec,
            mask=mask,
        )

    def separate(
        self,
        mixture: mx.array,
        query_mel: mx.array,
    ) -> BanquetOutput:
        """Separate audio using query.

        Convenience method that encodes the query and runs separation.

        Args:
            mixture: Mixture audio [batch, channels, samples]
            query_mel: Query mel spectrogram [batch, 1, n_mels, time]

        Returns:
            BanquetOutput with separated audio, spectrogram, and mask
        """
        # Encode query
        query_embedding = self.encode_query(query_mel)

        # Run separation
        return self(mixture, query_embedding)

    @staticmethod
    def from_config(
        config: BanquetConfig,
        passt_config: PaSSTConfig | None = None,
    ) -> Banquet:
        """Create Banquet from configuration.

        Args:
            config: Banquet model configuration
            passt_config: Optional PaSST configuration

        Returns:
            Banquet model
        """
        return Banquet(config=config, passt_config=passt_config)
