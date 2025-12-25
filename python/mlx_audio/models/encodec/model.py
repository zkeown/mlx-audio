"""EnCodec neural audio codec model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.models.encodec.config import EnCodecConfig
from mlx_audio.models.encodec.layers.decoder import EnCodecDecoder
from mlx_audio.models.encodec.layers.encoder import EnCodecEncoder
from mlx_audio.models.encodec.layers.quantizer import ResidualVectorQuantizer

if TYPE_CHECKING:
    pass


class EnCodec(nn.Module):
    """EnCodec neural audio codec.

    EnCodec compresses audio into discrete tokens using:
    1. Convolutional encoder to extract latent embeddings
    2. Residual vector quantization to discretize embeddings
    3. Convolutional decoder to reconstruct audio

    The codec can operate at different bitrates by using different
    numbers of codebooks (bandwidth selection).

    Attributes:
        config: Model configuration
        encoder: Convolutional encoder
        quantizer: Residual vector quantizer
        decoder: Convolutional decoder

    Example:
        >>> config = EnCodecConfig.encodec_32khz()
        >>> codec = EnCodec(config)
        >>> audio = mx.random.normal((1, 1, 32000))  # 1 second of audio
        >>> codes = codec.encode(audio)
        >>> reconstructed = codec.decode(codes)
    """

    def __init__(self, config: EnCodecConfig | None = None):
        """Initialize EnCodec model.

        Args:
            config: Model configuration. If None, uses 32kHz mono config.
        """
        super().__init__()
        if config is None:
            config = EnCodecConfig.encodec_32khz()
        self.config = config

        # Build encoder, quantizer, decoder
        self.encoder = EnCodecEncoder(config)
        self.quantizer = ResidualVectorQuantizer(
            num_codebooks=config.num_codebooks,
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
        )
        self.decoder = EnCodecDecoder(config)

    def encode(self, audio: mx.array) -> mx.array:
        """Encode audio waveform to discrete codes.

        Args:
            audio: Audio waveform with shape:
                   - [B, C, T]: Batched multichannel audio
                   - [B, T]: Batched mono audio
                   - [C, T]: Single multichannel audio
                   - [T]: Single mono audio

        Returns:
            Discrete codes [B, K, T'] where:
                - K is num_codebooks
                - T' is T / hop_length
        """
        # Handle different input shapes
        if audio.ndim == 1:
            audio = audio.reshape(1, 1, -1)  # [1, 1, T]
        elif audio.ndim == 2:
            if audio.shape[0] == self.config.channels:
                # [C, T] -> [1, C, T]
                audio = audio.reshape(1, *audio.shape)
            else:
                # [B, T] -> [B, 1, T]
                audio = audio.reshape(audio.shape[0], 1, audio.shape[1])
        # Now audio is [B, C, T]

        # Encode to latent embeddings
        embeddings = self.encoder(audio)  # [B, T', D]

        # Quantize to discrete codes
        _, codes = self.quantizer(embeddings)  # [B, K, T']

        return codes

    def decode(self, codes: mx.array) -> mx.array:
        """Decode discrete codes to audio waveform.

        Args:
            codes: Discrete codes [B, K, T'] where:
                   - K is num_codebooks
                   - T' is number of frames

        Returns:
            Audio waveform [B, C, T] where T = T' * hop_length
        """
        # Decode codes to embeddings
        embeddings = self.quantizer.decode(codes)  # [B, T', D]

        # Decode embeddings to audio
        audio = self.decoder(embeddings)  # [B, C, T]

        return audio

    def __call__(
        self,
        audio: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Encode and decode audio (full forward pass).

        Args:
            audio: Audio waveform [B, C, T] or [B, T]

        Returns:
            Tuple of:
                - Reconstructed audio [B, C, T]
                - Discrete codes [B, K, T']
        """
        codes = self.encode(audio)
        reconstructed = self.decode(codes)
        return reconstructed, codes

    @property
    def sample_rate(self) -> int:
        """Audio sample rate."""
        return self.config.sample_rate

    @property
    def channels(self) -> int:
        """Number of audio channels."""
        return self.config.channels

    @property
    def frame_rate(self) -> float:
        """Number of codec frames per second of audio."""
        return self.config.frame_rate

    @property
    def hop_length(self) -> int:
        """Total downsampling factor."""
        return self.config.hop_length

    @property
    def num_codebooks(self) -> int:
        """Number of codebooks."""
        return self.config.num_codebooks

    @property
    def codebook_size(self) -> int:
        """Size of each codebook."""
        return self.config.codebook_size

    def get_codebook(self, layer_idx: int) -> mx.array:
        """Get codebook weights for a specific layer.

        Args:
            layer_idx: Index of the codebook layer (0 to num_codebooks-1)

        Returns:
            Codebook weights [codebook_size, codebook_dim]
        """
        return self.quantizer.get_codebook(layer_idx)

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        **kwargs,
    ) -> EnCodec:
        """Load pretrained EnCodec model.

        Args:
            path: Path to model directory containing:
                  - config.json
                  - model.safetensors (or weights.npz)
            **kwargs: Additional arguments to override config

        Returns:
            Loaded EnCodec model
        """
        path = Path(path)

        # Load config
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            config = EnCodecConfig.from_dict(config_dict)
        else:
            # Default config
            config = EnCodecConfig.encodec_32khz()

        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Create model
        model = cls(config)

        # Load weights
        weights_path = path / "model.safetensors"
        if weights_path.exists():
            model.load_weights(str(weights_path))
        else:
            # Try .npz format
            npz_path = path / "weights.npz"
            if npz_path.exists():
                model.load_weights(str(npz_path))

        return model

    def save_pretrained(self, path: str | Path) -> None:
        """Save model to directory.

        Args:
            path: Output directory
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = path / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save weights
        weights_path = path / "model.safetensors"
        self.save_weights(str(weights_path))
