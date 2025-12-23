"""MusicGen text-to-music generation model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.models.musicgen.config import MusicGenConfig
from mlx_audio.models.musicgen.layers.embeddings import CodebookEmbeddings
from mlx_audio.models.musicgen.layers.transformer import MusicGenDecoder
from mlx_audio.models.musicgen.layers.lm_head import MusicGenLMHead, DelayPatternScheduler

if TYPE_CHECKING:
    from mlx_audio.models.encodec import EnCodec


class MusicGen(nn.Module):
    """MusicGen text-to-music generation model.

    MusicGen generates music from text descriptions using:
    1. T5 text encoder for conditioning
    2. Transformer decoder with delay pattern for multi-codebook generation
    3. EnCodec for audio tokenization/detokenization

    The model uses a delay pattern where different codebooks are offset
    in time, allowing parallel generation while respecting dependencies.

    Attributes:
        config: Model configuration
        embeddings: Codebook token embeddings
        decoder: Transformer decoder
        lm_head: Language model head for each codebook
        delay_pattern: Delay pattern scheduler

    Example:
        >>> config = MusicGenConfig.small()
        >>> model = MusicGen(config)
        >>> # Generate from text conditioning
        >>> text_embeds = model.encode_text(["jazz piano"])
        >>> codes = model.generate(text_embeds, duration=5.0)
    """

    def __init__(self, config: MusicGenConfig | None = None):
        """Initialize MusicGen model.

        Args:
            config: Model configuration. If None, uses small config.
        """
        super().__init__()
        if config is None:
            config = MusicGenConfig.small()
        self.config = config

        # Codebook embeddings
        self.embeddings = CodebookEmbeddings(config)

        # Text conditioning projection (T5 hidden size -> decoder hidden size)
        self.text_projection = nn.Linear(
            config.text_hidden_size,
            config.hidden_size,
        )

        # Transformer decoder
        self.decoder = MusicGenDecoder(config)

        # Language model head (one per codebook)
        self.lm_head = MusicGenLMHead(config)

        # Delay pattern scheduler
        self.delay_pattern = DelayPatternScheduler(
            num_codebooks=config.num_codebooks,
            pad_token_id=config.pad_token_id,
        )

        # Audio codec (loaded separately)
        self._audio_codec: "EnCodec" | None = None

    @property
    def audio_codec(self) -> "EnCodec":
        """Get audio codec (lazy loaded)."""
        if self._audio_codec is None:
            raise RuntimeError(
                "Audio codec not set. Use model.set_audio_codec() or load with from_pretrained()."
            )
        return self._audio_codec

    def set_audio_codec(self, codec: "EnCodec") -> None:
        """Set the audio codec for encoding/decoding audio.

        Args:
            codec: EnCodec model instance
        """
        self._audio_codec = codec

    def project_text_embeddings(
        self,
        text_embeddings: mx.array,
    ) -> mx.array:
        """Project text embeddings to decoder hidden size.

        Args:
            text_embeddings: Text encoder outputs [B, S, text_hidden_size]

        Returns:
            Projected embeddings [B, S, hidden_size]
        """
        return self.text_projection(text_embeddings)

    def forward(
        self,
        input_ids: mx.array,
        encoder_hidden_states: mx.array | None = None,
        attention_mask: mx.array | None = None,
        encoder_attention_mask: mx.array | None = None,
        kv_cache: list[tuple[mx.array, mx.array]] | None = None,
        position_ids: mx.array | None = None,
    ) -> tuple[mx.array, list[tuple[mx.array, mx.array]]]:
        """Forward pass through the model.

        Args:
            input_ids: Codebook token IDs [B, K, T]
            encoder_hidden_states: Text conditioning [B, S, D]
            attention_mask: Causal self-attention mask
            encoder_attention_mask: Cross-attention mask
            kv_cache: Cached key/values for incremental decoding
            position_ids: Position IDs [B, T]

        Returns:
            Tuple of:
                - Logits [B, K, T, V] for each codebook
                - Updated KV cache
        """
        # Compute embeddings
        hidden_states = self.embeddings(input_ids, position_ids)

        # Project text conditioning if provided
        if encoder_hidden_states is not None:
            encoder_hidden_states = self.project_text_embeddings(encoder_hidden_states)

        # Create causal mask if not provided
        if attention_mask is None:
            seq_length = input_ids.shape[-1]
            offset = 0
            if kv_cache is not None and len(kv_cache) > 0 and kv_cache[0] is not None:
                offset = kv_cache[0][0].shape[1]
            attention_mask = self.decoder.create_causal_mask(seq_length, offset)

        # Run decoder
        hidden_states, new_kv_cache = self.decoder(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            kv_cache=kv_cache,
        )

        # Compute logits for all codebooks
        logits = self.lm_head(hidden_states)  # [B, K, T, V]

        return logits, new_kv_cache

    def __call__(
        self,
        input_ids: mx.array,
        encoder_hidden_states: mx.array | None = None,
        **kwargs,
    ) -> mx.array:
        """Forward pass returning logits only.

        Args:
            input_ids: Codebook token IDs [B, K, T]
            encoder_hidden_states: Text conditioning [B, S, D]
            **kwargs: Additional arguments passed to forward()

        Returns:
            Logits [B, K, T, V]
        """
        logits, _ = self.forward(
            input_ids,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )
        return logits

    def generate(
        self,
        encoder_hidden_states: mx.array,
        max_new_tokens: int | None = None,
        duration: float | None = None,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        cfg_scale: float = 3.0,
        seed: int | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> mx.array:
        """Generate audio tokens from text conditioning.

        Uses autoregressive generation with the delay pattern to produce
        tokens for all codebooks.

        Args:
            encoder_hidden_states: Text conditioning [B, S, D]
            max_new_tokens: Maximum tokens to generate (overrides duration)
            duration: Generation duration in seconds
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter (0 to disable)
            top_p: Nucleus sampling threshold (0 to disable)
            cfg_scale: Classifier-free guidance scale (1.0 = no CFG)
            seed: Random seed for reproducibility
            progress_callback: Optional callback for progress updates

        Returns:
            Generated codes [B, K, T] where K is num_codebooks
        """
        from mlx_audio.models.musicgen.inference import generate_tokens

        # Determine max tokens
        if max_new_tokens is None:
            if duration is not None:
                max_new_tokens = int(duration * self.config.frame_rate)
            else:
                max_new_tokens = self.config.max_new_tokens

        # Generate tokens
        codes = generate_tokens(
            model=self,
            encoder_hidden_states=encoder_hidden_states,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            cfg_scale=cfg_scale,
            seed=seed,
            progress_callback=progress_callback,
        )

        return codes

    def decode_audio(self, codes: mx.array) -> mx.array:
        """Decode audio tokens to waveform.

        Args:
            codes: Audio codes [B, K, T]

        Returns:
            Audio waveform [B, C, samples]
        """
        return self.audio_codec.decode(codes)

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        **kwargs,
    ) -> "MusicGen":
        """Load pretrained MusicGen model.

        Args:
            path: Path to model directory containing:
                  - config.json
                  - model.safetensors (or weights.npz)
                  - Optionally: encodec/ subdirectory with codec weights
            **kwargs: Additional arguments to override config

        Returns:
            Loaded MusicGen model
        """
        path = Path(path)

        # Load config
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            config = MusicGenConfig.from_dict(config_dict)
        else:
            config = MusicGenConfig.small()

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
            npz_path = path / "weights.npz"
            if npz_path.exists():
                model.load_weights(str(npz_path))

        # Try to load audio codec if present
        encodec_path = path / "encodec"
        if encodec_path.exists():
            from mlx_audio.models.encodec import EnCodec
            codec = EnCodec.from_pretrained(encodec_path)
            model.set_audio_codec(codec)

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

        # Save audio codec if set
        if self._audio_codec is not None:
            encodec_path = path / "encodec"
            self._audio_codec.save_pretrained(encodec_path)
