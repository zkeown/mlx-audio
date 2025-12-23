"""Whisper speech recognition model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.models.whisper.config import WhisperConfig
from mlx_audio.models.whisper.layers.encoder import AudioEncoder
from mlx_audio.models.whisper.layers.decoder import TextDecoder

if TYPE_CHECKING:
    from mlx_audio.models.whisper.kv_cache import KVCache


class Whisper(nn.Module):
    """Whisper speech recognition model.

    Whisper is an encoder-decoder transformer trained on multilingual
    speech recognition, translation, language identification, and
    voice activity detection tasks.

    Architecture:
        - Encoder: Log-mel spectrogram -> Conv1d -> Transformer -> Audio features
        - Decoder: Tokens -> Transformer (with cross-attention) -> Logits

    The model supports:
        - Speech transcription (multilingual)
        - Speech translation (to English)
        - Language identification
        - Timestamp prediction

    Attributes:
        encoder: Audio encoder
        decoder: Text decoder
        config: Model configuration
    """

    def __init__(self, config: WhisperConfig | None = None):
        """Initialize Whisper model.

        Args:
            config: Model configuration. If None, uses default (tiny).
        """
        super().__init__()
        if config is None:
            config = WhisperConfig.tiny()
        self.config = config

        # Build encoder and decoder
        self.encoder = AudioEncoder(config)
        self.decoder = TextDecoder(config)

    def encode(self, mel: mx.array) -> mx.array:
        """Encode audio to features.

        Args:
            mel: Log-mel spectrogram [B, n_mels, T] or [n_mels, T]

        Returns:
            Audio features [B, T//2, n_state]
        """
        return self.encoder(mel)

    def decode(
        self,
        tokens: mx.array,
        audio_features: mx.array,
        kv_cache: "KVCache | None" = None,
    ) -> mx.array:
        """Decode tokens to logits.

        Args:
            tokens: Input token IDs [B, T]
            audio_features: Encoder output [B, S, D]
            kv_cache: Pre-allocated KV cache for efficient incremental
                decoding. Updated in-place.

        Returns:
            Logits [B, T, n_vocab]
        """
        return self.decoder(tokens, audio_features, kv_cache)

    def __call__(
        self,
        mel: mx.array,
        tokens: mx.array,
    ) -> mx.array:
        """Full forward pass.

        Args:
            mel: Log-mel spectrogram [B, n_mels, T]
            tokens: Input token IDs [B, L]

        Returns:
            Logits [B, L, n_vocab]
        """
        audio_features = self.encode(mel)
        logits = self.decode(tokens, audio_features)
        return logits

    def detect_language(
        self,
        mel: mx.array,
        tokenizer: "WhisperTokenizer",
    ) -> tuple[str, float]:
        """Detect the language of the audio.

        Args:
            mel: Log-mel spectrogram [B, n_mels, T] or [n_mels, T]
            tokenizer: Whisper tokenizer

        Returns:
            Tuple of (language_code, probability)
        """
        # Encode audio
        audio_features = self.encode(mel)

        # Get initial tokens (just SOT)
        sot_token = mx.array([[tokenizer.sot]])

        # Get logits for next token (no cache needed for single token)
        logits = self.decode(sot_token, audio_features)

        # Get probabilities over language tokens
        lang_tokens = tokenizer.all_language_tokens
        lang_logits = logits[0, 0, lang_tokens]
        lang_probs = mx.softmax(lang_logits, axis=-1)

        # Find best language
        best_idx = int(mx.argmax(lang_probs))
        best_prob = float(lang_probs[best_idx])

        # Get language code from token
        from mlx_audio.models.whisper.tokenizer import LANGUAGES
        lang_codes = list(LANGUAGES.keys())
        best_lang = lang_codes[best_idx]

        return best_lang, best_prob

    @property
    def dims(self) -> dict[str, int]:
        """Get model dimensions."""
        return {
            "n_mels": self.config.n_mels,
            "n_audio_ctx": self.config.n_audio_ctx,
            "n_audio_state": self.config.n_audio_state,
            "n_audio_head": self.config.n_audio_head,
            "n_audio_layer": self.config.n_audio_layer,
            "n_text_ctx": self.config.n_text_ctx,
            "n_text_state": self.config.n_text_state,
            "n_text_head": self.config.n_text_head,
            "n_text_layer": self.config.n_text_layer,
            "n_vocab": self.config.n_vocab,
        }

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        **kwargs,
    ) -> "Whisper":
        """Load pretrained Whisper model.

        Args:
            path: Path to model directory containing:
                  - config.json
                  - model.safetensors (or weights.npz)
            **kwargs: Additional arguments to override config

        Returns:
            Loaded Whisper model
        """
        path = Path(path)

        # Load config
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            config = WhisperConfig.from_dict(config_dict)
        else:
            # Try to infer from model name
            config = WhisperConfig.tiny()

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
