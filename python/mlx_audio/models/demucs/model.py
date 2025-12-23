"""HTDemucs model implementation for MLX.

Matches PyTorch demucs.htdemucs.HTDemucs exactly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.models.demucs.config import HTDemucsConfig
from mlx_audio.models.demucs.layers import (
    CrossTransformerEncoder,
    HDecLayer,
    HEncLayer,
)


class ScaledEmbedding(nn.Module):
    """Scaled embedding matching PyTorch's ScaledEmbedding.

    Has an inner embedding with key 'embedding.weight'.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        scale: float = 1.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.scale = scale

    def __call__(self, x: mx.array) -> mx.array:
        return self.embedding(x) * self.scale


class HTDemucs(nn.Module):
    """Hybrid Transformer Demucs model for source separation.

    Structure matches PyTorch exactly:
        encoder: list of HEncLayer (frequency branch, Conv2d)
        decoder: list of HDecLayer (frequency branch, ConvTranspose2d)
        tencoder: list of HEncLayer (time branch, Conv1d)
        tdecoder: list of HDecLayer (time branch, ConvTranspose1d)
        freq_emb: ScaledEmbedding
        channel_upsampler: Conv1d
        channel_downsampler: Conv1d
        channel_upsampler_t: Conv1d
        channel_downsampler_t: Conv1d
        crosstransformer: CrossTransformerEncoder
    """

    def __init__(self, config: HTDemucsConfig | None = None):
        super().__init__()
        if config is None:
            config = HTDemucsConfig()
        self.config = config

        # Calculate channel progression
        channels = [config.channels]
        for _ in range(config.depth - 1):
            channels.append(int(channels[-1] * config.growth))

        self.channels = channels
        chin = config.audio_channels
        chin_z = chin * 2 if config.cac else chin

        # Frequency branch encoders (encoder.0, encoder.1, ...)
        self.encoder = []
        for i, chout in enumerate(channels):
            chin_enc = chin_z if i == 0 else channels[i - 1]
            self.encoder.append(
                HEncLayer(
                    chin_enc, chout,
                    kernel_size=config.kernel_size,
                    stride=config.stride,
                    freq=True,
                    dconv_depth=config.dconv_depth,
                    dconv_compress=config.dconv_comp,
                )
            )

        # Time branch encoders (tencoder.0, tencoder.1, ...)
        self.tencoder = []
        for i, chout in enumerate(channels):
            chin_enc = chin if i == 0 else channels[i - 1]
            self.tencoder.append(
                HEncLayer(
                    chin_enc, chout,
                    kernel_size=config.kernel_size,
                    stride=config.stride,
                    freq=False,
                    dconv_depth=config.dconv_depth,
                    dconv_compress=config.dconv_comp,
                )
            )

        # Frequency branch decoders (decoder.0, decoder.1, ...)
        self.decoder = []
        for i in range(len(channels) - 1, -1, -1):
            chin_dec = channels[i]
            if i > 0:
                chout_dec = channels[i - 1]
            else:
                chout_dec = chin_z * config.num_sources
            self.decoder.append(
                HDecLayer(
                    chin_dec, chout_dec,
                    kernel_size=config.kernel_size,
                    stride=config.stride,
                    freq=True,
                    dconv_depth=config.dconv_depth,
                    dconv_compress=config.dconv_comp,
                    last=(i == 0),
                )
            )

        # Time branch decoders (tdecoder.0, tdecoder.1, ...)
        self.tdecoder = []
        for i in range(len(channels) - 1, -1, -1):
            chin_dec = channels[i]
            if i > 0:
                chout_dec = channels[i - 1]
            else:
                chout_dec = chin * config.num_sources
            self.tdecoder.append(
                HDecLayer(
                    chin_dec, chout_dec,
                    kernel_size=config.kernel_size,
                    stride=config.stride,
                    freq=False,
                    dconv_depth=config.dconv_depth,
                    dconv_compress=config.dconv_comp,
                    last=(i == 0),
                )
            )

        # Frequency embedding
        # Created after first encoder: nfft//2 -> nfft//2//stride freqs
        # Uses first encoder output channels (channels[0])
        # PyTorch ScaledEmbedding has internal scale=10 (for learning rate boost)
        # and external freq_emb_scale=0.2 (applied after embedding)
        self._n_freqs = (config.nfft // 2) // config.stride
        self.freq_emb = ScaledEmbedding(
            self._n_freqs, channels[0], scale=10.0  # Internal scale matches PyTorch
        )
        # Cache for precomputed frequency embeddings (set on first forward pass)
        self._cached_freq_emb: mx.array | None = None

        # Channel up/downsamplers for bridging encoder and transformer
        encoder_channels = channels[-1]  # Output of last encoder layer
        transformer_channels = config.bottom_channels or encoder_channels

        # Upsample: encoder -> transformer dimension
        self.channel_upsampler = nn.Conv1d(
            encoder_channels, transformer_channels, kernel_size=1
        )
        self.channel_downsampler = nn.Conv1d(
            transformer_channels, encoder_channels, kernel_size=1
        )
        self.channel_upsampler_t = nn.Conv1d(
            encoder_channels, transformer_channels, kernel_size=1
        )
        self.channel_downsampler_t = nn.Conv1d(
            transformer_channels, encoder_channels, kernel_size=1
        )

        # Cross-domain transformer
        self.crosstransformer = CrossTransformerEncoder(
            dim=transformer_channels,
            depth=config.t_depth,
            heads=config.t_heads,
            dim_feedforward=int(transformer_channels * config.t_hidden_scale),
            dropout=config.t_dropout,
        )

    def __call__(self, mix: mx.array) -> mx.array:
        """Forward pass.

        Args:
            mix: Input mixture [B, C, T] where C=audio_channels

        Returns:
            Separated stems [B, S, C, T] where S=num_sources
        """
        B, C, T = mix.shape
        S = self.config.num_sources

        # Pad to training length if needed (PyTorch use_train_segment behavior)
        # training_length = segment * samplerate
        training_length = int(self.config.segment * self.config.samplerate)
        length_pre_pad = None
        if T < training_length:
            length_pre_pad = T
            pad_amount = training_length - T
            mix = mx.pad(mix, [(0, 0), (0, 0), (0, pad_amount)])
            T = training_length

        # Compute STFT for frequency branch
        spec = self._compute_stft(mix)

        # Prepare frequency input - convert complex to real BEFORE normalization
        # PyTorch normalizes the real representation, not the complex values
        if self.config.cac:
            # Convert complex [B, C, F, T] to real [B, C*2, F, T]
            # PyTorch uses view_as_real().permute(0, 1, 4, 2, 3).reshape()
            # This interleaves: [real_ch0, imag_ch0, real_ch1, imag_ch1, ...]
            real_part = mx.real(spec)  # [B, C, F, T]
            imag_part = mx.imag(spec)  # [B, C, F, T]
            # Stack and reshape to interleave
            # [B, C, F, T] -> stack -> [B, C, 2, F, T] -> reshape [B, C*2, F, T]
            stacked = mx.stack([real_part, imag_part], axis=2)  # [B, C, 2, F, T]
            B_s, C_s, _, F_s, T_s = stacked.shape
            mag = stacked.reshape(B_s, C_s * 2, F_s, T_s)
        else:
            mag = mx.abs(spec)

        # Normalize frequency input (over all spatial dims like PyTorch)
        # PyTorch: x.mean(dim=(1, 2, 3)), x.std(dim=(1, 2, 3))
        spec_mean = mx.mean(mag, axis=(1, 2, 3), keepdims=True)
        spec_std = mx.std(mag, axis=(1, 2, 3), keepdims=True) + 1e-5
        freq_in = (mag - spec_mean) / spec_std

        # Normalize time input
        # PyTorch: xt.mean(dim=(1, 2)), xt.std(dim=(1, 2))
        mix_mean = mx.mean(mix, axis=(1, 2), keepdims=True)
        mix_std = mx.std(mix, axis=(1, 2), keepdims=True) + 1e-5
        mix_norm = (mix - mix_mean) / mix_std

        # Encode frequency branch
        # Store input lengths BEFORE encoding (for decoder trimming)
        # Store skips AFTER encoding
        freq_lengths = []
        freq_skips = []
        x = freq_in
        for idx, enc in enumerate(self.encoder):
            freq_lengths.append(x.shape[-1])  # Store input time dim
            x = enc(x)

            # Add frequency embedding after first encoder (PyTorch: idx == 0)
            if idx == 0 and hasattr(self, "freq_emb"):
                # x shape: [B, C, F, T]
                # Use cached embedding to avoid recomputing each forward pass
                if self._cached_freq_emb is None:
                    # First forward: compute and cache the frequency embedding
                    frs = mx.arange(self._n_freqs)  # Freq indices
                    emb = self.freq_emb(frs)  # [F, C]
                    self._cached_freq_emb = emb.T[None, :, :, None]  # [1, C, F, 1]
                x = x + self.config.freq_emb * self._cached_freq_emb

            freq_skips.append(x)

        # Encode time branch
        # Store input lengths BEFORE encoding (for decoder trimming)
        # Store skips AFTER encoding
        time_lengths = []
        time_skips = []
        xt = mix_norm
        for enc in self.tencoder:
            time_lengths.append(xt.shape[-1])  # Store input time dim
            xt = enc(xt)
            time_skips.append(xt)

        # Channel upsample before transformer
        # PyTorch pattern: flatten -> upsample -> unflatten -> transformer -> flatten -> downsample -> unflatten
        Bx, Cx, Fx, Tx = x.shape

        # Flatten freq: [B, C, F, T] -> [B, C, F*T]
        x_flat = x.reshape(Bx, Cx, Fx * Tx)

        # Upsample channels: Conv1d needs NLC format
        x_flat = x_flat.transpose(0, 2, 1)  # [B, F*T, C]
        x_flat = self.channel_upsampler(x_flat)
        x_flat = x_flat.transpose(0, 2, 1)  # [B, C', F*T]

        # Unflatten back to 4D for transformer: [B, C', F*T] -> [B, C', F, T]
        x = x_flat.reshape(Bx, -1, Fx, Tx)

        # Upsample time channels
        xt = xt.transpose(0, 2, 1)
        xt = self.channel_upsampler_t(xt)
        xt = xt.transpose(0, 2, 1)

        # Cross-domain transformer: freq [B, C', F, T], time [B, C', T']
        x, xt = self.crosstransformer(x, xt)

        # Flatten freq for downsample: [B, C', F, T] -> [B, C', F*T]
        x_flat = x.reshape(Bx, -1, Fx * Tx)

        # Downsample channels
        x_flat = x_flat.transpose(0, 2, 1)
        x_flat = self.channel_downsampler(x_flat)
        x_flat = x_flat.transpose(0, 2, 1)

        xt = xt.transpose(0, 2, 1)
        xt = self.channel_downsampler_t(xt)
        xt = xt.transpose(0, 2, 1)

        # Unflatten back to 4D: [B, C, F*T] -> [B, C, F, T]
        x = x_flat.reshape(Bx, Cx, Fx, Tx)

        # Decode frequency branch (with skips, reversed order)
        # Pass lengths in reverse order for decoder trimming
        for dec, skip, length in zip(
            self.decoder, reversed(freq_skips), reversed(freq_lengths)
        ):
            x, pre = dec(x, skip, length)

        # Decode time branch (with skips, reversed order)
        # Pass lengths in reverse order for decoder trimming
        for dec, skip, length in zip(
            self.tdecoder, reversed(time_skips), reversed(time_lengths)
        ):
            xt, pre = dec(xt, skip, length)

        # Process frequency branch output
        # x shape: [B, S*C*2, F, T] for CAC (each source outputs C*2 channels for real/imag)
        # Reshape to [B, S, C*2, F, T]
        F_out = x.shape[2]  # Number of freq bins
        T_spec = x.shape[3]  # Time frames
        x = x.reshape(B, S, C * 2 if self.config.cac else C, F_out, T_spec)

        # Denormalize frequency output: x = x * std + mean
        # spec_std/spec_mean are [B, 1, 1, 1], need to broadcast to [B, S, C*2, F, T]
        x = x * spec_std[:, None, ...] + spec_mean[:, None, ...]

        # Apply mask to original STFT and compute iSTFT
        freq_out = self._mask_and_istft(spec, x, T)

        # Process time branch output
        # xt shape: [B, S*C, T_out]
        T_out = xt.shape[-1]
        time_out = xt.reshape(B, S, C, T_out)

        # Pad or trim to original length
        if T_out < T:
            pad_amount = T - T_out
            time_out = mx.pad(time_out, [(0, 0), (0, 0), (0, 0), (0, pad_amount)])
        elif T_out > T:
            time_out = time_out[:, :, :, :T]

        # Denormalize time output
        time_out = time_out * mix_std[..., None] + mix_mean[..., None]

        # Combine frequency and time outputs (PyTorch: x = xt + x)
        output = time_out + freq_out

        # Trim to original length if we padded
        if length_pre_pad is not None:
            output = output[:, :, :, :length_pre_pad]

        return output

    def _compute_stft(self, x: mx.array) -> mx.array:
        """Compute STFT for frequency branch.

        Matches PyTorch demucs:
        - Uses reflect padding to align time frames
        - normalized=True (divides by sqrt(n_fft))
        - Removes Nyquist bin (last freq bin) to get n_fft//2 bins
        - Trims time frames to match expected output length
        """
        from mlx_audio.primitives import stft
        import math

        B, C, T = x.shape
        n_fft = self.config.nfft
        hop_length = self.config.hop_length

        # Calculate expected output length (matches PyTorch demucs padding)
        le = int(math.ceil(T / hop_length))
        pad = hop_length // 2 * 3

        # Pad input like PyTorch demucs
        x_padded = mx.pad(x, [(0, 0), (0, 0), (pad, pad + le * hop_length - T)])

        x_flat = x_padded.reshape(B * C, -1)
        spec = stft(x_flat, n_fft=n_fft, hop_length=hop_length, pad_mode="reflect")

        # Apply normalization (PyTorch normalized=True divides by sqrt(n_fft))
        spec = spec / math.sqrt(n_fft)

        # Remove Nyquist bin (last freq bin) to get n_fft//2 bins
        spec = spec[:, :-1, :]

        # Trim time frames: [2:2+le] to match PyTorch
        spec = spec[:, :, 2:2 + le]

        F, Tf = spec.shape[1], spec.shape[2]
        spec = spec.reshape(B, C, F, Tf)

        return spec

    def _compute_istft(self, spec: mx.array, length: int) -> mx.array:
        """Compute inverse STFT."""
        from mlx_audio.primitives import istft

        B, C, F, Tf = spec.shape
        hop_length = self.config.hop_length

        spec_flat = spec.reshape(B * C, F, Tf)
        x = istft(spec_flat, hop_length=hop_length, length=length)
        x = x.reshape(B, C, -1)

        return x

    def _mask_and_istft(
        self, z: mx.array, m: mx.array, length: int
    ) -> mx.array:
        """Apply mask to STFT and compute iSTFT.

        For CAC mode, m is the full spectrogram (not a mask), and z is ignored.

        Args:
            z: Original STFT [B, C, F, T] (complex)
            m: Decoder output [B, S, C*2, F, T] (real representation of complex)
            length: Target output length

        Returns:
            Audio output [B, S, C, T]
        """
        import math
        from mlx_audio.primitives import istft

        B, S, C_real, F, Tf = m.shape
        C = C_real // 2  # C*2 -> C (separate real/imag)
        hop_length = self.config.hop_length

        if self.config.cac:
            # CAC mode: m is the full spectrogram in real format
            # Convert from real [B, S, C*2, F, T] to complex [B, S, C, F, T]
            # The layout is [real_ch0, imag_ch0, real_ch1, imag_ch1, ...]
            # Reshape to [B, S, C, 2, F, T] then view as complex

            # [B, S, C*2, F, T] -> [B, S, C, 2, F, T]
            m_reshaped = m.reshape(B, S, C, 2, F, Tf)

            # Permute to [B, S, C, F, T, 2] for complex construction
            m_perm = m_reshaped.transpose(0, 1, 2, 4, 5, 3)

            # Construct complex: real + j*imag
            z_out = m_perm[..., 0] + 1j * m_perm[..., 1]  # [B, S, C, F, T]
        else:
            # Magnitude mask mode: apply mask to original STFT
            z_expanded = z[:, None, ...]  # [B, 1, C, F, T]
            z_out = (z_expanded / (1e-8 + mx.abs(z_expanded))) * m

        # Compute iSTFT for each source
        # z_out: [B, S, C, F, T]
        # Need to pad freq axis (add Nyquist bin) and time axis

        # Pad freq axis: add one freq bin (Nyquist)
        z_out = mx.pad(z_out, [(0, 0), (0, 0), (0, 0), (0, 1), (0, 0)])

        # Pad time axis: add 2 on each side (to match PyTorch)
        z_out = mx.pad(z_out, [(0, 0), (0, 0), (0, 0), (0, 0), (2, 2)])

        # Calculate output length with padding
        pad = hop_length // 2 * 3
        le = hop_length * int(math.ceil(length / hop_length)) + 2 * pad

        # Flatten batch and sources for iSTFT
        # [B, S, C, F, T] -> [B*S*C, F, T]
        z_flat = z_out.reshape(B * S * C, z_out.shape[3], z_out.shape[4])

        # Compute iSTFT
        x = istft(z_flat, hop_length=hop_length, length=le)

        # Apply normalization factor (PyTorch uses normalized=True which multiplies by sqrt(n_fft))
        n_fft = self.config.nfft
        x = x * math.sqrt(n_fft)

        # Reshape and trim
        # [B*S*C, T] -> [B, S, C, T]
        x = x.reshape(B, S, C, -1)

        # Trim padding: [pad:pad+length]
        x = x[:, :, :, pad:pad + length]

        return x

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        **kwargs: Any,
    ) -> "HTDemucs":
        """Load pretrained HTDemucs model."""
        path = Path(path)

        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            config = HTDemucsConfig.from_dict(config_dict)
        else:
            config = HTDemucsConfig()

        model = cls(config)

        weights_path = path / "model.safetensors"
        if weights_path.exists():
            model.load_weights(str(weights_path))
        else:
            for pattern in ["*.safetensors", "*.npz"]:
                matches = list(path.glob(pattern))
                if matches:
                    model.load_weights(str(matches[0]))
                    break

        return model

    def save_pretrained(self, path: str | Path) -> None:
        """Save model to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        config_dict = {
            "sources": self.config.sources,
            "audio_channels": self.config.audio_channels,
            "samplerate": self.config.samplerate,
            "segment": self.config.segment,
            "channels": self.config.channels,
            "growth": self.config.growth,
            "depth": self.config.depth,
            "kernel_size": self.config.kernel_size,
            "stride": self.config.stride,
            "nfft": self.config.nfft,
            "hop_length": self.config.hop_length,
            "freq_emb": self.config.freq_emb,
            "t_depth": self.config.t_depth,
            "t_heads": self.config.t_heads,
            "t_dropout": self.config.t_dropout,
            "t_hidden_scale": self.config.t_hidden_scale,
            "dconv_depth": self.config.dconv_depth,
            "cac": self.config.cac,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        self.save_weights(str(path / "model.safetensors"))
