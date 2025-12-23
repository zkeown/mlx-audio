"""CLAP (Contrastive Language-Audio Pretraining) model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.models.clap.config import CLAPConfig, CLAPAudioConfig, CLAPTextConfig
from mlx_audio.models.clap.feature_extractor import CLAPFeatureExtractor
from mlx_audio.models.clap.layers.htsat import HTSAT, AudioFusion
from mlx_audio.models.clap.layers.text_encoder import CLAPTextEncoder

if TYPE_CHECKING:
    import numpy as np


class CLAPProjection(nn.Module):
    """2-layer MLP projection head used by HuggingFace CLAP.

    Args:
        in_dim: Input dimension
        out_dim: Output projection dimension
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim, bias=True)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear1(x)
        x = nn.relu(x)
        x = self.linear2(x)
        return x


class CLAP(nn.Module):
    """CLAP model for audio-text embeddings.

    CLAP (Contrastive Language-Audio Pretraining) learns a joint embedding
    space for audio and text, enabling:
    - Audio similarity search
    - Zero-shot audio classification
    - Audio-text retrieval

    Architecture:
        - Audio encoder: HTSAT (Hierarchical Token-Semantic Audio Transformer)
        - Text encoder: RoBERTa-base
        - Shared projection space: 512 dimensions
        - Learnable temperature for similarity scaling

    Args:
        config: CLAP configuration
    """

    def __init__(self, config: CLAPConfig | None = None):
        super().__init__()
        if config is None:
            config = CLAPConfig()
        self.config = config

        # Audio encoder
        self.audio_encoder = HTSAT(config.audio)
        # Projection input is the HTSAT output dim (embed_dim doubled per stage)
        audio_encoder_dim = self.audio_encoder._final_dim
        self.audio_projection = CLAPProjection(
            audio_encoder_dim,
            config.projection_dim,
        )

        # Text encoder
        self.text_encoder = CLAPTextEncoder(config.text, config.projection_dim)

        # Fusion for variable-length audio
        if config.audio.enable_fusion:
            self.audio_fusion = AudioFusion(config.audio)
        else:
            self.audio_fusion = None

        # Learnable temperature (logit scale)
        self.logit_scale = mx.array([config.logit_scale_init])

        # Feature extractor for audio preprocessing
        self.feature_extractor = CLAPFeatureExtractor.from_config(config.audio)

    def encode_audio(
        self,
        audio: mx.array,
        normalize: bool = True,
        is_longer: mx.array | None = None,
    ) -> mx.array:
        """Encode audio to embeddings.

        Args:
            audio: Mel spectrogram [B, C, F, T] where C=1 or 4 for fusion
                   or [B, F, T] or waveform [B, T]
            normalize: L2 normalize embeddings
            is_longer: Boolean tensor [B] indicating which samples need fusion

        Returns:
            Audio embeddings [B, projection_dim]
        """
        # Handle different input shapes
        if audio.ndim == 2:
            # Assume waveform [B, T] - needs mel conversion
            audio, is_longer = self._compute_mel(audio)
        elif audio.ndim == 3:
            # Add channel dim: [B, F, T] -> [B, 1, F, T]
            audio = audio[:, None, :, :]

        # Use fusion for long audio
        if self.audio_fusion is not None and audio.shape[-1] > 1024:
            embeddings = self.audio_fusion(audio, self.audio_encoder)
        else:
            embeddings = self.audio_encoder(audio, is_longer=is_longer)

        # Project to shared space
        embeddings = self.audio_projection(embeddings)

        if normalize:
            embeddings = embeddings / mx.linalg.norm(embeddings, axis=-1, keepdims=True)

        return embeddings

    def encode_text(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        normalize: bool = True,
    ) -> mx.array:
        """Encode text to embeddings.

        Args:
            input_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L] (1=keep, 0=mask)
            normalize: L2 normalize embeddings

        Returns:
            Text embeddings [B, projection_dim]
        """
        embeddings = self.text_encoder(
            input_ids,
            attention_mask=attention_mask,
            normalize=normalize,
        )
        return embeddings

    def similarity(
        self,
        audio_embeds: mx.array,
        text_embeds: mx.array,
    ) -> mx.array:
        """Compute audio-text similarity matrix.

        Args:
            audio_embeds: Audio embeddings [B_a, dim]
            text_embeds: Text embeddings [B_t, dim]

        Returns:
            Similarity matrix [B_a, B_t]
        """
        # Scale by learned temperature
        scale = mx.exp(self.logit_scale)
        return scale * (audio_embeds @ text_embeds.T)

    def forward(
        self,
        audio: mx.array | None = None,
        input_ids: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ) -> dict[str, mx.array]:
        """Full forward pass for training.

        Args:
            audio: Mel spectrogram [B, 1, F, T]
            input_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L]

        Returns:
            Dictionary with audio_embeds, text_embeds, and logits
        """
        result = {}

        if audio is not None:
            audio_embeds = self.encode_audio(audio, normalize=True)
            result["audio_embeds"] = audio_embeds

        if input_ids is not None:
            text_embeds = self.encode_text(
                input_ids,
                attention_mask=attention_mask,
                normalize=True,
            )
            result["text_embeds"] = text_embeds

        if "audio_embeds" in result and "text_embeds" in result:
            result["logits_per_audio"] = self.similarity(
                result["audio_embeds"],
                result["text_embeds"],
            )
            result["logits_per_text"] = result["logits_per_audio"].T

        return result

    def __call__(
        self,
        audio: mx.array | None = None,
        input_ids: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ) -> dict[str, mx.array]:
        """Forward pass."""
        return self.forward(audio, input_ids, attention_mask)

    def _compute_mel(self, audio: mx.array) -> tuple[mx.array, mx.array]:
        """Compute log-mel spectrogram from waveform using HF-compatible preprocessing.

        Args:
            audio: Waveform [B, T]

        Returns:
            Tuple of:
                - Log-mel spectrogram [B, C, F, T] where C=1 or 4 for fusion
                - is_longer: Boolean tensor [B, 1] indicating which samples need fusion
        """
        # Use the feature extractor for HF-compatible preprocessing
        result = self.feature_extractor(audio, return_tensors="mx")
        mel = result["input_features"]  # [B, C, T, F]
        is_longer = result["is_longer"]  # [B, 1]

        # Transpose from [B, C, T, F] to [B, C, F, T] for HTSAT
        mel = mx.transpose(mel, (0, 1, 3, 2))

        return mel, is_longer

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        **kwargs,
    ) -> "CLAP":
        """Load pretrained CLAP model.

        Args:
            path: Path to model directory containing:
                  - config.json
                  - model.safetensors
            **kwargs: Additional arguments to override config

        Returns:
            Loaded CLAP model
        """
        path = Path(path)

        # Check if weights need conversion (PyTorch -> MLX)
        weights_path = path / "model.safetensors"
        needs_conversion = False
        if weights_path.exists():
            needs_conversion = cls._check_needs_conversion(weights_path)

        if needs_conversion:
            # Convert PyTorch weights to MLX format
            path = cls._convert_weights(path)

        # Load config
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            config = CLAPConfig.from_dict(config_dict)
        else:
            config = CLAPConfig()

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

        # Set to eval mode for inference (uses running stats in BatchNorm)
        model.eval()

        return model

    @staticmethod
    def _check_needs_conversion(weights_path: Path) -> bool:
        """Check if weights need PyTorch -> MLX conversion."""
        from safetensors import safe_open
        with safe_open(str(weights_path), framework="np") as f:
            keys = list(f.keys())
            # PyTorch weights have audio_model. prefix, MLX has audio_encoder.
            return any(k.startswith("audio_model.") for k in keys)

    @classmethod
    def _convert_weights(cls, source_path: Path) -> Path:
        """Convert PyTorch weights to MLX format."""
        from mlx_audio.models.clap.convert import convert_clap_weights

        # Create output path in cache directory
        cache_dir = Path.home() / ".cache" / "mlx_audio"
        model_name = source_path.name + "-mlx"
        output_path = cache_dir / model_name

        if (output_path / "model.safetensors").exists():
            # Already converted
            return output_path

        print(f"Converting CLAP weights from {source_path} to MLX format...")

        # Use the HuggingFace model ID if available
        # The source_path might be a HF cache path
        try:
            convert_clap_weights(str(source_path), output_path)
        except Exception:
            # Try with the original repo ID
            convert_clap_weights("laion/clap-htsat-fused", output_path)

        return output_path

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
