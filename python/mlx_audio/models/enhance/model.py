"""DeepFilterNet speech enhancement model."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import DeepFilterNetConfig
from .layers.erb import erb_filterbank, erb_inverse_filterbank
from .layers.grouped import GroupedGRU, GroupedLinear

if TYPE_CHECKING:
    pass


class ErbEncoder(nn.Module):
    """ERB-domain encoder for spectral envelope estimation."""

    def __init__(self, config: DeepFilterNetConfig):
        super().__init__()
        self.config = config

        # Input: ERB features
        # GRU layers for temporal modeling
        self.gru = nn.GRU(
            config.erb_bands,
            config.hidden_size,
            num_layers=config.enc_layers,
        )

        # Hidden to ERB gains
        self.fc = nn.Linear(config.hidden_size, config.erb_bands)

    def __call__(
        self,
        erb_feat: mx.array,
        hidden: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Encode ERB features.

        Parameters
        ----------
        erb_feat : mx.array
            ERB features of shape (batch, n_frames, erb_bands).
        hidden : mx.array, optional
            Initial hidden state.

        Returns
        -------
        tuple
            (erb_gains, hidden) where erb_gains is (batch, n_frames, erb_bands).
        """
        # GRU encoding
        enc_out, new_hidden = self.gru(erb_feat, hidden)

        # Predict ERB gains (sigmoid for 0-1 range)
        erb_gains = mx.sigmoid(self.fc(enc_out))

        return erb_gains, new_hidden


class DfDecoder(nn.Module):
    """Deep filter decoder for fine-grained enhancement."""

    def __init__(self, config: DeepFilterNetConfig):
        super().__init__()
        self.config = config
        self.df_order = config.df_order
        self.df_bins = config.df_bins

        # GRU for temporal modeling
        self.gru = nn.GRU(
            config.hidden_size + config.df_bins * 2,  # Hidden + complex spec
            config.hidden_size,
            num_layers=1,
        )

        # Predict filter coefficients: df_order * df_bins * 2 (real/imag)
        self.fc = nn.Linear(
            config.hidden_size,
            config.df_order * config.df_bins * 2,
        )

    def __call__(
        self,
        hidden_state: mx.array,
        spec_feat: mx.array,
        df_hidden: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Predict deep filter coefficients.

        Parameters
        ----------
        hidden_state : mx.array
            Encoder hidden state (batch, n_frames, hidden_size).
        spec_feat : mx.array
            Complex spectrogram features (batch, n_frames, df_bins * 2).
        df_hidden : mx.array, optional
            Deep filter GRU hidden state.

        Returns
        -------
        tuple
            (df_coefs, new_hidden) where df_coefs is
            (batch, n_frames, df_order, df_bins, 2).
        """
        batch, n_frames, _ = hidden_state.shape

        # Concatenate hidden state with spectrogram features
        combined = mx.concatenate([hidden_state, spec_feat], axis=-1)

        # GRU processing
        out, new_hidden = self.gru(combined, df_hidden)

        # Predict coefficients
        coefs = self.fc(out)  # (batch, n_frames, df_order * df_bins * 2)

        # Reshape to (batch, n_frames, df_order, df_bins, 2)
        coefs = coefs.reshape(
            batch, n_frames, self.df_order, self.df_bins, 2
        )

        return coefs, new_hidden


class DeepFilterNet(nn.Module):
    """DeepFilterNet speech enhancement model.

    Two-stage architecture:
    1. ERB stage: Enhances spectral envelope using ERB-scaled gains
    2. DF stage: Deep filtering for periodic component enhancement

    Parameters
    ----------
    config : DeepFilterNetConfig
        Model configuration.
    """

    def __init__(self, config: DeepFilterNetConfig | None = None):
        super().__init__()
        self.config = config or DeepFilterNetConfig()

        # ERB filterbanks (precomputed, not trainable)
        erb_fb = erb_filterbank(
            self.config.fft_size,
            self.config.sample_rate,
            self.config.erb_bands,
        )
        erb_inv = erb_inverse_filterbank(
            self.config.fft_size,
            self.config.sample_rate,
            self.config.erb_bands,
        )
        self._erb_fb = erb_fb
        self._erb_inv = erb_inv

        # ERB encoder
        self.erb_encoder = ErbEncoder(self.config)

        # Deep filter decoder
        self.df_decoder = DfDecoder(self.config)

    def _stft(self, audio: mx.array) -> mx.array:
        """Compute STFT of audio.

        Parameters
        ----------
        audio : mx.array
            Audio waveform (batch, samples).

        Returns
        -------
        mx.array
            Complex STFT (batch, n_frames, n_freqs).
        """
        from mlx_audio.primitives import stft

        # STFT
        spec = stft(
            audio,
            n_fft=self.config.fft_size,
            hop_length=self.config.hop_size,
            win_length=self.config.frame_size,
            window="hann",
            center=True,
        )

        # Transpose to (batch, n_frames, n_freqs)
        if spec.ndim == 2:
            spec = spec[None, :]
        spec = mx.transpose(spec, (0, 2, 1))

        return spec

    def _istft(self, spec: mx.array, length: int) -> mx.array:
        """Compute inverse STFT.

        Parameters
        ----------
        spec : mx.array
            Complex STFT (batch, n_frames, n_freqs).
        length : int
            Output audio length.

        Returns
        -------
        mx.array
            Audio waveform (batch, samples).
        """
        from mlx_audio.primitives import istft

        # Transpose to (batch, n_freqs, n_frames)
        spec = mx.transpose(spec, (0, 2, 1))

        # iSTFT
        audio = istft(
            spec,
            hop_length=self.config.hop_size,
            win_length=self.config.frame_size,
            window="hann",
            center=True,
            length=length,
        )

        return audio

    def _apply_erb_gains(
        self,
        spec_mag: mx.array,
        erb_gains: mx.array,
    ) -> mx.array:
        """Apply ERB gains to magnitude spectrogram.

        Parameters
        ----------
        spec_mag : mx.array
            Magnitude spectrogram (batch, n_frames, n_freqs).
        erb_gains : mx.array
            ERB gains (batch, n_frames, erb_bands).

        Returns
        -------
        mx.array
            Enhanced magnitude (batch, n_frames, n_freqs).
        """
        # Expand ERB gains to full frequency resolution
        # erb_gains: (batch, n_frames, erb_bands)
        # erb_inv: (n_freqs, erb_bands)
        gains_full = erb_gains @ self._erb_inv.T  # (batch, n_frames, n_freqs)

        return spec_mag * gains_full

    def _apply_deep_filter(
        self,
        spec: mx.array,
        coefs: mx.array,
    ) -> mx.array:
        """Apply deep filter to complex spectrogram.

        Parameters
        ----------
        spec : mx.array
            Complex spectrogram (batch, n_frames, n_freqs).
        coefs : mx.array
            Filter coefficients (batch, n_frames, df_order, df_bins, 2).

        Returns
        -------
        mx.array
            Filtered spectrogram (batch, n_frames, n_freqs).
        """
        batch, n_frames, n_freqs = spec.shape
        df_bins = self.config.df_bins
        df_order = self.config.df_order

        # Extract low-frequency bins for deep filtering
        spec_low = spec[:, :, :df_bins]  # (batch, n_frames, df_bins)

        # Convert coefficients to complex
        coefs_complex = coefs[..., 0] + 1j * coefs[..., 1]
        # Shape: (batch, n_frames, df_order, df_bins)

        # Apply filter: convolve along time axis
        # For simplicity, use center-aligned convolution
        half_order = df_order // 2
        filtered = mx.zeros_like(spec_low)

        for t in range(n_frames):
            for k in range(df_order):
                t_src = t - half_order + k
                if 0 <= t_src < n_frames:
                    filtered[:, t, :] += spec_low[:, t_src, :] * coefs_complex[:, t, k, :]

        # Combine with high frequencies (unchanged)
        spec_out = mx.concatenate([filtered, spec[:, :, df_bins:]], axis=-1)

        return spec_out

    def __call__(
        self,
        audio: mx.array,
        erb_hidden: mx.array | None = None,
        df_hidden: mx.array | None = None,
    ) -> mx.array:
        """Enhance audio.

        Parameters
        ----------
        audio : mx.array
            Input audio (batch, samples) or (samples,).
        erb_hidden : mx.array, optional
            ERB encoder hidden state (for streaming).
        df_hidden : mx.array, optional
            Deep filter hidden state (for streaming).

        Returns
        -------
        mx.array
            Enhanced audio with same shape as input.
        """
        # Handle 1D input
        input_is_1d = audio.ndim == 1
        if input_is_1d:
            audio = audio[None, :]

        n_samples = audio.shape[-1]

        # Compute STFT
        spec = self._stft(audio)  # (batch, n_frames, n_freqs)

        # Get magnitude and phase
        spec_mag = mx.abs(spec)
        spec_phase = mx.angle(spec) if hasattr(mx, 'angle') else np.angle(np.array(spec))
        if isinstance(spec_phase, np.ndarray):
            spec_phase = mx.array(spec_phase.astype(np.float32))

        # Compute ERB features
        # erb_fb: (erb_bands, n_freqs)
        erb_feat = spec_mag @ self._erb_fb.T  # (batch, n_frames, erb_bands)

        # Log compression
        erb_feat = mx.log(erb_feat + 1e-8)

        # ERB encoding and gain prediction
        erb_gains, new_erb_hidden = self.erb_encoder(erb_feat, erb_hidden)

        # Apply ERB gains
        spec_mag_erb = self._apply_erb_gains(spec_mag, erb_gains)

        # Prepare features for deep filter
        df_bins = self.config.df_bins
        spec_low = spec[:, :, :df_bins]
        spec_feat = mx.concatenate([
            mx.real(spec_low) if hasattr(mx, 'real') else spec_low.real,
            mx.imag(spec_low) if hasattr(mx, 'imag') else spec_low.imag,
        ], axis=-1)

        # Get encoder hidden for DF decoder
        # Use last hidden state from GRU
        enc_hidden = new_erb_hidden
        if enc_hidden.ndim == 3:
            # (num_layers, batch, hidden) -> (batch, n_frames, hidden)
            enc_hidden_expanded = mx.broadcast_to(
                enc_hidden[-1:].transpose((1, 0, 2)),
                (spec.shape[0], spec.shape[1], self.config.hidden_size)
            )
        else:
            enc_hidden_expanded = mx.broadcast_to(
                enc_hidden[None, :],
                (spec.shape[0], spec.shape[1], self.config.hidden_size)
            )

        # Deep filter prediction and application
        df_coefs, new_df_hidden = self.df_decoder(
            enc_hidden_expanded, spec_feat, df_hidden
        )
        spec_df = self._apply_deep_filter(
            spec_mag_erb * mx.exp(1j * spec_phase),
            df_coefs,
        )

        # Inverse STFT
        enhanced = self._istft(spec_df, n_samples)

        if input_is_1d:
            enhanced = enhanced[0]

        return enhanced

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path | None = None,
        config: DeepFilterNetConfig | None = None,
    ) -> "DeepFilterNet":
        """Load pretrained model.

        Parameters
        ----------
        path : str or Path, optional
            Path to model weights. If None, uses random initialization.
        config : DeepFilterNetConfig, optional
            Model configuration. If None and path is provided, loads from path.

        Returns
        -------
        DeepFilterNet
            Loaded model.
        """
        if config is None:
            config = DeepFilterNetConfig.deepfilternet2()

        model = cls(config)

        if path is not None:
            path = Path(path)
            weights_path = path / "weights.npz" if path.is_dir() else path

            if weights_path.exists():
                weights = mx.load(str(weights_path))
                model.load_weights(list(weights.items()))

        return model
